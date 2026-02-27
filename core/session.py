"""
Gestionnaire d'état de la session de diagnostic CIF.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EtapeDiagnostic(Enum):
    ACCUEIL = "accueil"
    COLLECTE_DESCRIPTION = "collecte_description"
    IDENTIFICATION_FICHE = "identification_fiche"
    COLLECTE_NIVEAU1 = "collecte_niveau1"       # Open questions phase
    COMPLEMENTS = "complements"                  # "Anything else?" loop
    VALIDATION = "validation"                    # Customer validates final summary
    TERMINE = "termine"


@dataclass
class SessionDiagnostic:
    etape: EtapeDiagnostic = EtapeDiagnostic.ACCUEIL

    description_initiale: str = ""

    # Identified fault
    domaine_id: Optional[str] = None
    domaine_nom: Optional[str] = None
    fiche_id: Optional[str] = None
    fiche_titre: Optional[str] = None

    # Question counter (safety cap)
    compteur_questions: int = 0

    # Full conversation history (sent to LLM)
    historique: list = field(default_factory=list)

    # Running summary — updated after each complement
    synthese: Optional[str] = None

    def ajouter_message(self, role: str, contenu: str):
        self.historique.append({"role": role, "content": contenu})

    def to_dict(self) -> dict:
        return {
            "etape": self.etape.value,
            "description_initiale": self.description_initiale,
            "perimeter": self.domaine_nom,
            "cif_title": self.fiche_titre,
            "questions_asked": self.compteur_questions,
            "synthese": self.synthese,
        }
