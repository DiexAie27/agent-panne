import os, uuid, json
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
    session.repair_order_id = body.repair_order_id
    session.vehicle = body.vehicle
    session.customer_name = body.customer_name
    sessions[token] = session
    base_url = os.environ.get("FLUTTER_APP_URL", "https://single-repairer-portal-fwp8v2.flutterflow.app/")
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
    return {
        "response": response,
        "etape": session.etape.value,
        "termine": session.etape == EtapeDiagnostic.TERMINE,
        "synthese": session.synthese if session.etape == EtapeDiagnostic.TERMINE else None
    }
