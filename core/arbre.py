"""
Gestionnaire de l'arbre de décision CIF.
Structure : PERIMETER -> CIF TITLE -> LEVEL 1 -> LEVEL 2 -> LEVEL 3
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

    def domaines_pour_prompt(self) -> str:
        return "\n".join(f"- {d['nom']} (id: {d['id']})" for d in self.data["domaines"])

    def liste_fiches(self, domaine_id: str) -> list:
        domaine = self.domaines.get(domaine_id)
        if not domaine:
            return []
        return [{"id": f["id"], "titre": f["titre"]} for f in domaine["fiches"]]

    def fiches_pour_prompt(self, domaine_id: str) -> str:
        return "\n".join(f"- {f['titre']} (id: {f['id']})" for f in self.liste_fiches(domaine_id))

    def get_fiche(self, domaine_id: str, fiche_id: str) -> Optional[dict]:
        domaine = self.domaines.get(domaine_id)
        if not domaine:
            return None
        for f in domaine["fiches"]:
            if f["id"] == fiche_id:
                return f
        return None

    def get_niveau1_options(self, domaine_id: str, fiche_id: str) -> list:
        fiche = self.get_fiche(domaine_id, fiche_id)
        if not fiche:
            return []
        return [n["label"] for n in fiche["niveaux"]]

    def get_niveau2_options(self, domaine_id: str, fiche_id: str, niveau1_label: str) -> list:
        fiche = self.get_fiche(domaine_id, fiche_id)
        if not fiche:
            return []
        for n1 in fiche["niveaux"]:
            if n1["label"].strip() == niveau1_label.strip():
                return [n2["label"] for n2 in n1["options"]]
        return []

    def get_niveau3_options(self, domaine_id: str, fiche_id: str, niveau1_label: str, niveau2_label: str) -> list:
        fiche = self.get_fiche(domaine_id, fiche_id)
        if not fiche:
            return []
        for n1 in fiche["niveaux"]:
            if n1["label"].strip() == niveau1_label.strip():
                for n2 in n1["options"]:
                    if n2["label"].strip() == niveau2_label.strip():
                        return n2.get("options", [])
        return []

    def niveau1_pour_prompt(self, domaine_id: str, fiche_id: str) -> str:
        return "\n".join(f"- {o}" for o in self.get_niveau1_options(domaine_id, fiche_id))

    def niveau2_pour_prompt(self, domaine_id: str, fiche_id: str, niveau1: str) -> str:
        return "\n".join(f"- {o}" for o in self.get_niveau2_options(domaine_id, fiche_id, niveau1))

    def niveau3_pour_prompt(self, domaine_id: str, fiche_id: str, niveau1: str, niveau2: str) -> str:
        return "\n".join(f"- {o}" for o in self.get_niveau3_options(domaine_id, fiche_id, niveau1, niveau2))
