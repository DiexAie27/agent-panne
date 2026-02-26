"""
Agent conversationnel de diagnostic CIF - approche questions ouvertes.
Le LLM lit l'arbre en arrière-plan et pose jusqu'à 4 questions naturelles,
en sautant ce que le client a déjà dit. Restitution finale sans conseil de réparation.
"""

import os
import json
from groq import Groq
from core.session import SessionDiagnostic, EtapeDiagnostic
from core.arbre import ArbreDecision


# ------------------------------------------------------------------ #
#  Prompts                                                            #
# ------------------------------------------------------------------ #

PROMPT_SYSTEM_BASE = """You are a vehicle fault intake specialist. 
Your role is to collect a precise description of a vehicle fault from a customer, 
using natural conversational language — not technical jargon.

Key rules:
- Ask ONE question at a time, in plain language a non-expert can understand
- Never show the customer any internal tree structure, codes, or numbered lists from the diagnostic system
- Never suggest causes or fixes
- If the customer has already answered something in their description, do not ask it again
- Always respond in the same language the customer uses
"""

PROMPT_IDENTIFIER_FICHE = """
A customer described a vehicle fault as follows:
"{description}"

Below is the list of known fault types, grouped by system (perimeter):
{arbre_complet}

Your tasks:
1. Identify the single most likely perimeter (system) and CIF fault title that matches the description.
2. If you are not confident, identify the top 2 candidates.

Reply ONLY with a JSON object:
{{
  "perimeter": "exact perimeter name",
  "cif_title": "exact CIF title",
  "confiance": "high|medium|low",
  "question": "one natural language clarification question if confidence is low or medium, else null"
}}
"""

PROMPT_PROCHAINE_QUESTION = """
You are collecting information about a vehicle fault to fill a diagnostic report.

The fault has been identified as:
- Perimeter: {perimeter}
- CIF Title: {cif_title}

The diagnostic tree for this fault requires capturing the following dimensions:
{dimensions}

Conversation so far:
{historique}

Your task:
- Review what the customer has already said.
- Identify which dimensions are NOT yet clearly covered by their answers.
- If all dimensions are covered, reply with {{"action": "done"}}.
- If there are uncovered dimensions, ask the NEXT most important open question in plain, friendly language.
  Do not reveal the tree structure. Do not offer numbered options. Ask as a human would.

Reply ONLY with a JSON object:
{{
  "action": "ask" | "done",
  "question": "your natural language question, or null if done",
  "dimensions_manquantes": ["list of dimension labels still missing"]
}}
"""

PROMPT_SYNTHESE = """
Based on the conversation below, write a concise factual paragraph (5-8 lines) 
summarising the vehicle fault for a technician.

Rules:
- Use plain, factual language
- Do NOT suggest any cause, diagnosis, or repair
- Do NOT use bullet points — write a single flowing paragraph
- Capture all details the customer mentioned: what happens, when, any warning lights, 
  any messages on screen, any conditions that make it better or worse

Fault identification:
- Perimeter: {perimeter}
- CIF Title: {cif_title}

Conversation:
{historique}
"""


# ------------------------------------------------------------------ #
#  Formatage de l'arbre pour les prompts                             #
# ------------------------------------------------------------------ #

def _formater_arbre_pour_identification(arbre: ArbreDecision) -> str:
    """Retourne une version textuelle compacte de tous les domaines + titres CIF."""
    lignes = []
    for domaine in arbre.data["domaines"]:
        lignes.append(f"\n[{domaine['nom']}]")
        for fiche in domaine["fiches"]:
            lignes.append(f"  - {fiche['titre']}")
    return "\n".join(lignes)


def _formater_dimensions(arbre: ArbreDecision, domaine_id: str, fiche_id: str) -> str:
    """
    Extrait les dimensions clés de l'arbre pour une fiche donnée.
    Chaque 'dimension' est un axe d'observation que le LLM doit couvrir.
    """
    fiche = arbre.get_fiche(domaine_id, fiche_id)
    if not fiche:
        return ""

    lignes = []

    # Dimension 1 : les Level 1 = nature/catégorie du problème
    niveaux1 = [n["label"] for n in fiche["niveaux"]]
    lignes.append(f"Dimension 1 - Nature of the problem: {' | '.join(niveaux1)}")

    # Dimension 2 : les Level 2 = comportement/sévérité
    tous_niveaux2 = set()
    for n1 in fiche["niveaux"]:
        for n2 in n1["options"]:
            if isinstance(n2, dict):
                tous_niveaux2.add(n2["label"])
            elif isinstance(n2, str):
                tous_niveaux2.add(n2)
    if tous_niveaux2:
        lignes.append(f"Dimension 2 - Behaviour / severity: {' | '.join(sorted(tous_niveaux2))}")

    # Dimension 3 : les Level 3 = contexte observable (warning lamps, messages, workarounds)
    tous_niveaux3 = set()
    for n1 in fiche["niveaux"]:
        for n2 in n1["options"]:
            if isinstance(n2, dict):
                for n3 in n2.get("options", []):
                    tous_niveaux3.add(n3)
    if tous_niveaux3:
        # On regroupe par thème pour ne pas noyer le LLM
        lignes.append(f"Dimension 3 - Observable context (warning lamps, dashboard messages, workarounds): "
                      f"{' | '.join(sorted(tous_niveaux3))}")

    return "\n".join(lignes)


def _formater_historique(historique: list) -> str:
    """Formate l'historique pour injection dans un prompt."""
    lignes = []
    for msg in historique:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lignes.append(f"{role}: {msg['content']}")
    return "\n".join(lignes)


# ------------------------------------------------------------------ #
#  Agent principal                                                    #
# ------------------------------------------------------------------ #

class AgentDiagnostic:

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.arbre = ArbreDecision()
        self.model = "llama-3.3-70b-versatile"
        # Limite de questions ouvertes avant de passer à la synthèse
        self.max_questions = 4

    # ------------------------------------------------------------------ #
    #  Point d'entrée                                                     #
    # ------------------------------------------------------------------ #

    def traiter_message(self, message: str, session: SessionDiagnostic) -> str:
        if message:
            session.ajouter_message("user", message)

        if session.etape == EtapeDiagnostic.ACCUEIL:
            reponse = self._etape_accueil(session)

        elif session.etape == EtapeDiagnostic.COLLECTE_DESCRIPTION:
            reponse = self._etape_description(message, session)

        elif session.etape == EtapeDiagnostic.IDENTIFICATION_FICHE:
            # Le client a répondu à une question de clarification sur l'identification
            reponse = self._identifier_fiche(session)

        elif session.etape == EtapeDiagnostic.COLLECTE_NIVEAU1:
            # Phase principale : questions ouvertes pilotées par le LLM
            reponse = self._etape_questions_ouvertes(message, session)

        elif session.etape == EtapeDiagnostic.VALIDATION:
            reponse = self._etape_validation(message, session)

        else:
            reponse = "The session is complete. Thank you!"

        session.ajouter_message("assistant", reponse)
        return reponse

    # ------------------------------------------------------------------ #
    #  Étapes                                                             #
    # ------------------------------------------------------------------ #

    def _etape_accueil(self, session: SessionDiagnostic) -> str:
        session.etape = EtapeDiagnostic.COLLECTE_DESCRIPTION
        return (
            "Hello! I'm here to help you report a vehicle fault so our technical team "
            "can assist you as quickly as possible.\n\n"
            "Please describe the issue you're experiencing, in your own words."
        )

    def _etape_description(self, message: str, session: SessionDiagnostic) -> str:
        session.description_initiale = message
        session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
        return self._identifier_fiche(session)

    def _identifier_fiche(self, session: SessionDiagnostic) -> str:
        """
        Utilise le LLM pour identifier le périmètre et la fiche CIF.
        Si confiance faible, pose une question de clarification et reste dans cet état.
        Si confiance haute/moyenne, passe aux questions ouvertes.
        """
        arbre_texte = _formater_arbre_pour_identification(self.arbre)
        historique_texte = _formater_historique(session.historique)

        prompt = PROMPT_IDENTIFIER_FICHE.format(
            description=historique_texte,
            arbre_complet=arbre_texte,
        )
        result = self._llm_json(prompt)

        if not result:
            # Fallback neutre
            session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
            return "Could you tell me a bit more about what's happening with the vehicle?"

        # Retrouver les IDs correspondant aux noms retournés par le LLM
        domaine_id, fiche_id = self._resoudre_ids(
            result.get("perimeter", ""),
            result.get("cif_title", "")
        )

        if domaine_id and fiche_id:
            session.domaine_id = domaine_id
            session.domaine_nom = result["perimeter"]
            session.fiche_id = fiche_id
            session.fiche_titre = result["cif_title"]

            if result.get("confiance") == "low" and result.get("question"):
                # On reste en IDENTIFICATION_FICHE et on pose la question
                return result["question"]
            else:
                # Confiance suffisante : on passe aux questions ouvertes
                session.etape = EtapeDiagnostic.COLLECTE_NIVEAU1
                session.compteur_questions = 0
                return self._poser_prochaine_question(session)
        else:
            # Le LLM n'a pas trouvé de correspondance claire
            session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
            question = result.get("question") or "Can you describe more specifically what system or part of the vehicle is affected?"
            return question

    def _etape_questions_ouvertes(self, message: str, session: SessionDiagnostic) -> str:
        """
        Phase principale : le LLM décide si une nouvelle question est nécessaire
        ou si toutes les dimensions sont couvertes.
        """
        if not hasattr(session, 'compteur_questions'):
            session.compteur_questions = 0

        session.compteur_questions += 1

        # Sécurité : ne pas dépasser max_questions
        if session.compteur_questions >= self.max_questions:
            return self._generer_synthese(session)

        return self._poser_prochaine_question(session)

    def _poser_prochaine_question(self, session: SessionDiagnostic) -> str:
        """
        Demande au LLM quelle est la prochaine question à poser,
        ou s'il faut passer à la synthèse.
        """
        dimensions = _formater_dimensions(self.arbre, session.domaine_id, session.fiche_id)
        historique_texte = _formater_historique(session.historique)

        prompt = PROMPT_PROCHAINE_QUESTION.format(
            perimeter=session.domaine_nom,
            cif_title=session.fiche_titre,
            dimensions=dimensions,
            historique=historique_texte,
        )
        result = self._llm_json(prompt)

        if not result or result.get("action") == "done":
            return self._generer_synthese(session)

        question = result.get("question")
        if not question:
            return self._generer_synthese(session)

        return question

    def _generer_synthese(self, session: SessionDiagnostic) -> str:
        """Génère le paragraphe de synthèse et le soumet à validation."""
        historique_texte = _formater_historique(session.historique)

        prompt = PROMPT_SYNTHESE.format(
            perimeter=session.domaine_nom,
            cif_title=session.fiche_titre,
            historique=historique_texte,
        )
        synthese = self._llm_texte(prompt)
        session.synthese = synthese
        session.etape = EtapeDiagnostic.VALIDATION

        return (
            "Thank you for all the details. Here is the summary I'll send to our technical team:\n\n"
            f"---\n{synthese}\n---\n\n"
            "Does this accurately reflect the issue you're experiencing? "
            "Reply **yes** to confirm, or let me know what should be corrected."
        )

    def _etape_validation(self, message: str, session: SessionDiagnostic) -> str:
        positif = any(w in message.lower() for w in [
            "yes", "correct", "ok", "right", "confirm", "good", "perfect",
            "oui", "valide", "parfait", "correct", "c'est ça"
        ])
        if positif:
            session.etape = EtapeDiagnostic.TERMINE
            return (
                "Your fault report has been recorded. A technician will be in touch shortly. "
                "Thank you for your time."
            )
        else:
            # Le client veut corriger : on relance les questions ouvertes
            session.etape = EtapeDiagnostic.COLLECTE_NIVEAU1
            session.compteur_questions = 0
            return (
                "No problem. Could you tell me what was inaccurate or missing, "
                "and I'll update the report."
            )

    # ------------------------------------------------------------------ #
    #  Résolution des IDs depuis les noms retournés par le LLM           #
    # ------------------------------------------------------------------ #

    def _resoudre_ids(self, perimeter_nom: str, cif_titre: str):
        """
        Retrouve domaine_id et fiche_id à partir des noms en texte libre
        retournés par le LLM (correspondance insensible à la casse).
        """
        domaine_id = None
        fiche_id = None

        for domaine in self.arbre.data["domaines"]:
            if domaine["nom"].lower() == perimeter_nom.lower():
                domaine_id = domaine["id"]
                for fiche in domaine["fiches"]:
                    if fiche["titre"].lower() == cif_titre.lower():
                        fiche_id = fiche["id"]
                        break
                break

        # Fallback : correspondance partielle si exacte échoue
        if not domaine_id:
            for domaine in self.arbre.data["domaines"]:
                if perimeter_nom.lower() in domaine["nom"].lower():
                    domaine_id = domaine["id"]
                    for fiche in domaine["fiches"]:
                        if cif_titre.lower() in fiche["titre"].lower() or \
                           fiche["titre"].lower() in cif_titre.lower():
                            fiche_id = fiche["id"]
                            break
                    break

        return domaine_id, fiche_id

    # ------------------------------------------------------------------ #
    #  Appels LLM                                                         #
    # ------------------------------------------------------------------ #

    def _llm_json(self, prompt: str) -> dict | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM_BASE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            contenu = response.choices[0].message.content.strip()
            contenu = contenu.replace("```json", "").replace("```", "").strip()
            return json.loads(contenu)
        except Exception as e:
            print(f"[LLM JSON ERROR] {e}")
            return None

    def _llm_texte(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM_BASE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM TEXT ERROR] {e}")
            return "Error generating summary."
