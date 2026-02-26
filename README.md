# Agent de Diagnostic de Pannes

Agent conversationnel qui guide le client à travers un arbre de décision pour qualifier une panne.

## Structure du projet

```
agent-panne/
├── app.py                    # Interface Chainlit (point d'entrée)
├── requirements.txt          
├── .env.example              
├── data/
│   └── arbre_decision.json   # Arbre de décision (à personnaliser)
└── core/
    ├── agent.py              # Logique principale de l'agent
    ├── arbre.py              # Chargement et navigation de l'arbre
    └── session.py            # Gestion de l'état de la session
```

## Installation

```bash
# 1. Cloner / copier le projet
cd agent-panne

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API Groq (gratuit sur https://console.groq.com)
cp .env.example .env
# Éditer .env et renseigner votre GROQ_API_KEY

# 5. Lancer l'application
chainlit run app.py
```

L'interface s'ouvre automatiquement sur http://localhost:8000

## Flux de la conversation

```
ACCUEIL
  └─► COLLECTE_DESCRIPTION  (client décrit la panne)
        └─► IDENTIFICATION_DOMAINE  (LLM identifie le domaine)
              └─► IDENTIFICATION_SOUS_DOMAINE  (LLM affine)
                    └─► CONTEXTE_1  (question spécifique)
                          └─► CONTEXTE_2  (question spécifique)
                                └─► SYNTHESE  (paragraphe généré)
                                      └─► VALIDATION  (client valide ou corrige)
                                            └─► TERMINE
```

## Personnaliser l'arbre de décision

Éditez `data/arbre_decision.json` pour ajouter vos propres domaines, sous-domaines et questions contextuelles.

Structure minimale d'un domaine :
```json
{
  "id": "mon_domaine",
  "nom": "Mon Domaine",
  "description": "Description courte",
  "mots_cles": ["mot1", "mot2"],
  "sous_domaines": [
    {
      "id": "mon_sous_domaine",
      "nom": "Mon Sous-Domaine",
      "description": "Description",
      "mots_cles": ["mot3"],
      "contexte_1": {
        "question": "Ma question de contexte ?",
        "options": ["Option A", "Option B", "Option C"]
      },
      "contexte_2": {
        "question": "Ma deuxième question ?",
        "options": ["Option X", "Option Y"]
      }
    }
  ]
}
```
