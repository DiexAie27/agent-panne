"""
Gestionnaire de l'arbre de décision CIF - nouveau format avec EFC codes et conditions.
Structure : domaine -> fiche -> niveaux (L1>L2>L3 avec efc) + conditions (CAP)
"""

import json
from pathlib import Path
from typing import Optional


class ArbreDecision:

    def __init__(self, chemin_json: str = "data/arbre_cif.json"):
        chemin = Path(chemin_json)
        with open(chemin, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.domaines = {d["id"]: d for d in self.data["domaines"]}

    def liste_domaines(self) -> list:
        return [{"id": d["id"], "nom": d["nom"]} for d in self.data["domaines"]]

    def get_fiche(self, domaine_id: str, fiche_id: str) -> Optional[dict]:
        domaine = self.domaines.get(domaine_id)
        if not domaine:
            return None
        for f in domaine["fiches"]:
            if f["id"] == fiche_id:
                return f
        return None

    # ------------------------------------------------------------------ #
    #  Formatage pour identification (liste complète domaines + titres)  #
    # ------------------------------------------------------------------ #

    def arbre_pour_identification(self) -> str:
        """Compact text listing of all perimeters and CIF titles for the LLM."""
        lignes = []
        for domaine in self.data["domaines"]:
            lignes.append(f"\nPERIMETER: {domaine['nom']}")
            for fiche in domaine["fiches"]:
                lignes.append(f"  CIF: {fiche['titre']}")
        return "\n".join(lignes)

    # ------------------------------------------------------------------ #
    #  Dimensions pour PROMPT_PROCHAINE_QUESTION                         #
    # ------------------------------------------------------------------ #

    def dimensions_pour_prompt(self, domaine_id: str, fiche_id: str) -> str:
        """
        Builds the full dimensions string injected into PROMPT_PROCHAINE_QUESTION.
        Covers L1 / L2 / L3 symptom labels + CAP conditions of appearance.
        """
        fiche = self.get_fiche(domaine_id, fiche_id)
        if not fiche:
            return ""
        lignes = []

        # Dimension 1 — L1: nature of the problem
        niveaux1 = [n["label"] for n in fiche["niveaux"]]
        lignes.append(f"Dimension 1 - Nature of the problem: {' | '.join(niveaux1)}")

        # Dimension 2 — L2: behaviour / severity (all unique values)
        tous_niveaux2 = set()
        for n1 in fiche["niveaux"]:
            for n2 in n1.get("options", []):
                if isinstance(n2, dict):
                    tous_niveaux2.add(n2["label"])
        if tous_niveaux2:
            lignes.append(f"Dimension 2 - Behaviour / severity: {' | '.join(sorted(tous_niveaux2))}")

        # Dimension 3 — L3: observable context (warning lamps, messages, workarounds)
        tous_niveaux3 = set()
        for n1 in fiche["niveaux"]:
            for n2 in n1.get("options", []):
                if isinstance(n2, dict):
                    for n3 in n2.get("options", []):
                        if isinstance(n3, dict):
                            tous_niveaux3.add(n3["label"])
        if tous_niveaux3:
            lignes.append(
                f"Dimension 3 - Observable context (warning lamps, messages, workarounds): "
                f"{' | '.join(sorted(tous_niveaux3))}"
            )

        # Dimension 4 — CAP conditions of appearance
        conditions = fiche.get("conditions", [])
        if conditions:
            lignes.append("Dimension 4 - Conditions of appearance:")
            for cap in conditions:
                options_str = " | ".join(cap.get("options", []))
                lignes.append(f"  - {cap['categorie']}: {options_str}")

        return "\n".join(lignes)

    # ------------------------------------------------------------------ #
    #  Résolution IDs (recherche par nom texte libre)                    #
    # ------------------------------------------------------------------ #

    def resoudre_ids(self, perimeter_nom: str, cif_titre: str):
        """
        Finds (domaine_id, fiche_id) from free-text names returned by the LLM.
        Tries exact match first, then partial match.
        """
        for domaine in self.data["domaines"]:
            if domaine["nom"].lower() == perimeter_nom.lower():
                for fiche in domaine["fiches"]:
                    if fiche["titre"].lower() == cif_titre.lower():
                        return domaine["id"], fiche["id"]
                # Perimeter matched exactly — try partial fiche match
                for fiche in domaine["fiches"]:
                    if (cif_titre.lower() in fiche["titre"].lower() or
                            fiche["titre"].lower() in cif_titre.lower()):
                        return domaine["id"], fiche["id"]

        # Partial perimeter match
        for domaine in self.data["domaines"]:
            if (perimeter_nom.lower() in domaine["nom"].lower() or
                    domaine["nom"].lower() in perimeter_nom.lower()):
                for fiche in domaine["fiches"]:
                    if (cif_titre.lower() in fiche["titre"].lower() or
                            fiche["titre"].lower() in cif_titre.lower()):
                        return domaine["id"], fiche["id"]

        return None, None
