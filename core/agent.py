"""
Agent conversationnel de diagnostic CIF - approche questions ouvertes.
Flux : identification fiche → questions ouvertes → boucle compléments → validation.
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
2. If you are not confident, identify the top candidate and prepare a clarification question.

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
Based on the conversation below, write a concise fault summary paragraph (5-8 lines)
for a vehicle technician.

Rules:
- Write in a neutral, factual tone — like a technician's field note
- Do NOT mention the customer, use "the vehicle" or passive constructions instead
- Do NOT suggest any cause, diagnosis, or repair
- Do NOT use bullet points — write a single flowing paragraph
- Capture all observed details: what happens, under what conditions, any warning lights,
  any dashboard messages, anything that makes it better or worse
- If something was not observed or not present (e.g. no warning light), include that too

Example style:
"During braking, a whistling sound occurs systematically, accompanied by visible smoke.
The vehicle pulls to the right at the same time. The issue is consistent and reproducible
regardless of road conditions. No warning light is present on the dashboard."

Fault identification:
- Perimeter: {perimeter}
- CIF Title: {cif_title}

Conversation:
{historique}
"""

PROMPT_REFORMULER_COMPLEMENT = """
Below is the current fault summary and a new detail just provided.
Rewrite the summary to naturally incorporate the new information.

Rules:
- Write in a neutral, factual tone — like a technician's field note
- Do NOT mention the customer, use "the vehicle" or passive constructions instead
- Do NOT suggest any cause, diagnosis, or repair
- Do NOT use bullet points — write a single flowing paragraph
- Do NOT lose any previously captured information

Current summary:
{synthese_actuelle}

New detail:
{complement}

Write the updated paragraph directly, no preamble.
"""

# Phrases that mean "nothing more to add"
NEGATIVE_RESPONSES = [
    "no", "nope", "nothing", "that's all", "that's it", "nothing else",
    "no more", "i'm good", "done", "finish", "that's everything",
    "non", "rien", "c'est tout", "rien d'autre", "c'est bon", "terminé"
]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _formater_arbre_pour_identification(arbre: ArbreDecision) -> str:
    lignes = []
    for domaine in arbre.data["domaines"]:
        lignes.append(f"\n[{domaine['nom']}]")
        for fiche in domaine["fiches"]:
            lignes.append(f"  - {fiche['titre']}")
    return "\n".join(lignes)


def _formater_dimensions(arbre: ArbreDecision, domaine_id: str, fiche_id: str) -> str:
    fiche = arbre.get_fiche(domaine_id, fiche_id)
    if not fiche:
        return ""
    lignes = []
    niveaux1 = [n["label"] for n in fiche["niveaux"]]
    lignes.append(f"Dimension 1 - Nature of the problem: {' | '.join(niveaux1)}")
    tous_niveaux2 = set()
    for n1 in fiche["niveaux"]:
        for n2 in n1["options"]:
            if isinstance(n2, dict):
                tous_niveaux2.add(n2["label"])
            elif isinstance(n2, str):
                tous_niveaux2.add(n2)
    if tous_niveaux2:
        lignes.append(f"Dimension 2 - Behaviour / severity: {' | '.join(sorted(tous_niveaux2))}")
    tous_niveaux3 = set()
    for n1 in fiche["niveaux"]:
        for n2 in n1["options"]:
            if isinstance(n2, dict):
                for n3 in n2.get("options", []):
                    tous_niveaux3.add(n3)
    if tous_niveaux3:
        lignes.append(
            f"Dimension 3 - Observable context (warning lamps, dashboard messages, workarounds): "
            f"{' | '.join(sorted(tous_niveaux3))}"
        )
    return "\n".join(lignes)


def _formater_historique(historique: list) -> str:
    lignes = []
    for msg in historique:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lignes.append(f"{role}: {msg['content']}")
    return "\n".join(lignes)


def _est_reponse_negative(message: str) -> bool:
    """Détecte si le client dit qu'il n'a rien à ajouter."""
    msg = message.strip().lower().rstrip(".,!?")
    return msg in NEGATIVE_RESPONSES or any(msg == neg for neg in NEGATIVE_RESPONSES)


# ------------------------------------------------------------------ #
#  Agent principal                                                    #
# ------------------------------------------------------------------ #

class AgentDiagnostic:

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.arbre = ArbreDecision()
        self.model = "llama-3.3-70b-versatile"
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
            reponse = self._identifier_fiche(session)

        elif session.etape == EtapeDiagnostic.COLLECTE_NIVEAU1:
            reponse = self._etape_questions_ouvertes(message, session)

        elif session.etape == EtapeDiagnostic.COMPLEMENTS:
            reponse = self._etape_complements(message, session)

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
        arbre_texte = _formater_arbre_pour_identification(self.arbre)
        historique_texte = _formater_historique(session.historique)
        prompt = PROMPT_IDENTIFIER_FICHE.format(
            description=historique_texte,
            arbre_complet=arbre_texte,
        )
        result = self._llm_json(prompt)

        if not result:
            session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
            return "Could you tell me a bit more about what's happening with the vehicle?"

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
                return result["question"]
            else:
                session.etape = EtapeDiagnostic.COLLECTE_NIVEAU1
                session.compteur_questions = 0
                return self._poser_prochaine_question(session)
        else:
            session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
            return result.get("question") or "Can you describe more specifically what system or part of the vehicle is affected?"

    def _etape_questions_ouvertes(self, message: str, session: SessionDiagnostic) -> str:
        session.compteur_questions += 1
        if session.compteur_questions >= self.max_questions:
            return self._lancer_complements(session)
        return self._poser_prochaine_question(session)

    def _poser_prochaine_question(self, session: SessionDiagnostic) -> str:
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
            return self._lancer_complements(session)

        question = result.get("question")
        if not question:
            return self._lancer_complements(session)

        return question

    # ------------------------------------------------------------------ #
    #  Boucle compléments                                                 #
    # ------------------------------------------------------------------ #

    def _lancer_complements(self, session: SessionDiagnostic) -> str:
        """Génère une première synthèse et ouvre la boucle compléments."""
        session.synthese = self._appel_synthese(session)
        session.etape = EtapeDiagnostic.COMPLEMENTS
        return (
            "Thank you, I now have a good picture of the issue. "
            "Here is what I have captured so far:\n\n"
            f"---\n{session.synthese}\n---\n\n"
            "Is there anything else you have noticed or would like to add — "
            "even if it seems minor? For example, a noise, a smell, a specific condition "
            "when it happens, or anything unusual about the vehicle."
        )

    def _etape_complements(self, message: str, session: SessionDiagnostic) -> str:
        """
        Boucle : le client ajoute des informations jusqu'à dire 'non'.
        À chaque ajout, la synthèse est mise à jour et on redemande.
        """
        if _est_reponse_negative(message):
            # Le client n'a rien à ajouter → on passe à la validation
            return self._soumettre_synthese(session)

        # Le client a ajouté quelque chose → on met à jour la synthèse
        prompt = PROMPT_REFORMULER_COMPLEMENT.format(
            synthese_actuelle=session.synthese,
            complement=message,
        )
        session.synthese = self._llm_texte(prompt)

        # On redemande s'il y a autre chose
        return (
            "Noted, I've updated the report. Here is the revised summary:\n\n"
            f"---\n{session.synthese}\n---\n\n"
            "Is there anything else you'd like to add?"
        )

    def _soumettre_synthese(self, session: SessionDiagnostic) -> str:
        """Présente la synthèse finale au client pour validation."""
        session.etape = EtapeDiagnostic.VALIDATION
        return (
            "Here is the complete summary I'll send to our technical team:\n\n"
            f"---\n{session.synthese}\n---\n\n"
            "Does this accurately reflect everything you've described? "
            "Reply **yes** to confirm, or let me know what should be corrected."
        )

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    def _etape_validation(self, message: str, session: SessionDiagnostic) -> str:
        positif = any(w in message.lower() for w in [
            "yes", "correct", "ok", "right", "confirm", "good", "perfect",
            "oui", "valide", "parfait", "c'est ça"
        ])
        if positif:
            session.etape = EtapeDiagnostic.TERMINE
            return (
                "Your fault report has been recorded. A technician will be in touch shortly. "
                "Thank you for your time."
            )
        else:
            # Restart complement loop with the correction
            session.etape = EtapeDiagnostic.COMPLEMENTS
            return (
                "No problem. Could you tell me what was inaccurate or missing? "
                "I'll update the report."
            )

    # ------------------------------------------------------------------ #
    #  Génération de synthèse                                             #
    # ------------------------------------------------------------------ #

    def _appel_synthese(self, session: SessionDiagnostic) -> str:
        historique_texte = _formater_historique(session.historique)
        prompt = PROMPT_SYNTHESE.format(
            perimeter=session.domaine_nom,
            cif_title=session.fiche_titre,
            historique=historique_texte,
        )
        return self._llm_texte(prompt)

    # ------------------------------------------------------------------ #
    #  Résolution IDs                                                     #
    # ------------------------------------------------------------------ #

    def _resoudre_ids(self, perimeter_nom: str, cif_titre: str):
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
