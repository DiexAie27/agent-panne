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

Below is the list of known fault types. Format is:
  PERIMETER: <system category>
    CIF: <specific fault title>

{arbre_complet}

Your tasks:
1. Identify the single most likely PERIMETER and CIF fault title that matches the description.
2. The cif_title must be a value that appears after "CIF:" — never a PERIMETER name.
3. If you are not confident, prepare a clarification question.

Reply ONLY with a JSON object:
{{
  "perimeter": "exact PERIMETER name",
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
- Read the ENTIRE conversation carefully, including the very first message where the customer described their issue.
- Any information already mentioned — even in the first message — must be treated as KNOWN. Do NOT ask about it again.
- Identify only dimensions that are genuinely absent from everything said so far.
- If all key dimensions are covered, reply with {{"action": "done"}}.
- If there is a truly missing dimension, ask ONE question about it in plain, friendly language.
  Do not reveal the tree structure. Do not offer numbered options. Ask as a human would.

Reply ONLY with a JSON object:
{{
  "action": "ask" | "done",
  "question": "your natural language question, or null if done",
  "dimensions_manquantes": ["list of dimension labels genuinely missing"]
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

PROMPT_RANKING_CIF = """
Based on the full conversation below, identify the most probable CIF fault type(s)
from the list provided. For each, estimate a probability as a percentage.

The list below uses this format:
  PERIMETER: <system category name>
    CIF: <specific fault title>

You must return the PERIMETER name and the CIF title exactly as written after "CIF:".
Never return a PERIMETER name as a cif_title.

Rules:
- Always consider whether 2 or 3 CIF titles could plausibly match — do not default to 1 if there is genuine ambiguity
- If the description is precise and one CIF is clearly dominant: return 1 entry at 100%
- If the description could match 2 CIF titles: return 2 entries, split the probability honestly (e.g. 70/30)
- If the description is vague and 3 CIF titles are plausible: return 3 entries (e.g. 50/30/20)
- Probabilities must always sum to 100
- Be honest about uncertainty — it is better to show 2 candidates than to force a single wrong answer
- cif_title must ALWAYS be a value that appears after "CIF:" in the list — never a PERIMETER name

Known fault types:
{arbre_complet}

Conversation:
{historique}

Reply ONLY with a JSON array (1 to 3 items):
[
  {{"perimeter": "exact PERIMETER name", "cif_title": "exact CIF title", "probabilite": 70}},
  {{"perimeter": "exact PERIMETER name", "cif_title": "exact CIF title", "probabilite": 30}}
]
"""

# Phrases that mean "nothing more to add"
NEGATIVE_RESPONSES = [
    "no", "nope", "nothing", "that's all", "that's it", "nothing else",
    "no more", "i'm good", "done", "finish", "that's everything",
    "non", "rien", "c'est tout", "rien d'autre", "c'est bon", "terminé"
]


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _formater_historique(historique: list) -> str:
    lignes = []
    for msg in historique:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lignes.append(f"{role}: {msg['content']}")
    return "\n".join(lignes)


def _est_reponse_negative(message: str) -> bool:
    msg = message.strip().lower().rstrip(".,!?")
    return msg in NEGATIVE_RESPONSES


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
        historique_texte = _formater_historique(session.historique)
        prompt = PROMPT_IDENTIFIER_FICHE.format(
            description=historique_texte,
            arbre_complet=self.arbre.arbre_pour_identification(),
        )
        result = self._llm_json(prompt)

        if not result:
            session.etape = EtapeDiagnostic.IDENTIFICATION_FICHE
            return "Could you tell me a bit more about what's happening with the vehicle?"

        domaine_id, fiche_id = self.arbre.resoudre_ids(
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
        dimensions = self.arbre.dimensions_pour_prompt(session.domaine_id, session.fiche_id)
        historique_texte = _formater_historique(session.historique)

        # Prepend the initial description explicitly so the LLM cannot overlook it
        contexte = (
            f"[Initial description from customer]: {session.description_initiale}\n\n"
            f"{historique_texte}"
        )

        prompt = PROMPT_PROCHAINE_QUESTION.format(
            perimeter=session.domaine_nom,
            cif_title=session.fiche_titre,
            dimensions=dimensions,
            historique=contexte,
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
        if _est_reponse_negative(message):
            return self._soumettre_synthese(session)

        prompt = PROMPT_REFORMULER_COMPLEMENT.format(
            synthese_actuelle=session.synthese,
            complement=message,
        )
        session.synthese = self._llm_texte(prompt)

        return (
            "Noted, I've updated the report. Here is the revised summary:\n\n"
            f"---\n{session.synthese}\n---\n\n"
            "Is there anything else you'd like to add?"
        )

    def _soumettre_synthese(self, session: SessionDiagnostic) -> str:
        session.etape = EtapeDiagnostic.VALIDATION
        # Run CIF ranking
        ranking = self._appel_ranking_cif(session)
        session.cif_ranking = ranking
        ranking_texte = self._formater_ranking(ranking)
        return (
            "Here is the complete summary I'll send to our technical team:\n\n"
            f"---\n{session.synthese}\n---\n\n"
            f"{ranking_texte}\n\n"
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

    def _appel_ranking_cif(self, session: SessionDiagnostic) -> list:
        """Asks the LLM to rank the top 1-3 most probable CIF with probabilities."""
        historique_texte = _formater_historique(session.historique)
        prompt = PROMPT_RANKING_CIF.format(
            arbre_complet=self.arbre.arbre_pour_identification(),
            historique=historique_texte,
        )
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
            ranking = json.loads(contenu)

            # Validate: filter out entries where cif_title is actually a perimeter name
            perimeter_names = {d["nom"].lower() for d in self.arbre.data["domaines"]}
            valid = [
                item for item in ranking
                if item.get("cif_title", "").lower() not in perimeter_names
                and item.get("cif_title", "") != ""
            ]

            # If validation removed everything, fall back
            if not valid:
                raise ValueError("All ranking entries were invalid (perimeter names used as CIF titles)")

            return valid

        except Exception as e:
            print(f"[LLM RANKING ERROR] {e}")
            # Fallback: return the fiche identified during the conversation
            return [{"perimeter": session.domaine_nom,
                     "cif_title": session.fiche_titre,
                     "probabilite": 100}]

    def _formater_ranking(self, ranking: list) -> str:
        """Formats the CIF ranking as a readable block for the customer message."""
        if not ranking:
            return ""
        lignes = ["**Most probable fault type(s):**"]
        for item in ranking:
            prob = item.get("probabilite", "?")
            perimeter = item.get("perimeter", "")
            titre = item.get("cif_title", "")
            lignes.append(f"- {prob}% — {perimeter} / {titre}")
        return "\n".join(lignes)

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
