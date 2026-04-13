import os, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.agent import AgentDiagnostic
from core.session import SessionDiagnostic, EtapeDiagnostic

app = FastAPI()

# Required for FlutterFlow to call your API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AgentDiagnostic()

# Simple in-memory session store (fine for mockup)
sessions: dict[str, SessionDiagnostic] = {}


class MessageIn(BaseModel):
    session_id: str
    message: str = ""


class LinkRequest(BaseModel):
    repair_order_id: str
    vehicle: str = ""
    customer_name: str = ""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate-link")
def generate_link(body: LinkRequest):
    """Called by the SA to create a diagnostic session and get a shareable URL."""
    token = str(uuid.uuid4())
    session = SessionDiagnostic()
    # Store repair order context directly on the session object
    session.repair_order_id = body.repair_order_id
    session.vehicle = body.vehicle
    session.customer_name = body.customer_name
    sessions[token] = session
    base_url = os.environ.get("FLUTTER_APP_URL", "https://your-flutterflow-app.com")
    return {
        "token": token,
        "url": f"{base_url}/diagnostic?token={token}"
    }


@app.post("/chat")
def chat(body: MessageIn):
    """Called by the customer chat page on every message."""
    session = sessions.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    response = agent.traiter_message(body.message, session)

    termine = session.etape == EtapeDiagnostic.TERMINE

    return {
        # Agent reply to display in the chat
        "response": response,

        # Current step — use this in FlutterFlow to show/hide the finish button
        # Possible values: accueil | collecte_description | identification_fiche |
        #                  collecte_niveau1 | complements | validation | termine
        "etape": session.etape.value,

        # True only when the full diagnostic is complete and validated
        "termine": termine,

        # Free-text summary paragraph — available from the complements phase onward
        "synthese": session.synthese,

        # CIF probability ranking — populated only when termine = true
        # Format: [{"perimeter": str, "cif_title": str, "probabilite": int}, ...]
        "cif_ranking": session.cif_ranking if termine else [],

        # Identified fault reference (available once fiche is identified)
        "perimeter": session.domaine_nom,
        "cif_title": session.fiche_titre,
        "fiche_id": session.fiche_id,
    }
