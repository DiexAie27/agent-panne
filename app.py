"""
Interface Chainlit pour l'agent de diagnostic CIF.
Lancer avec : chainlit run app.py
"""

import chainlit as cl
from core.agent import AgentDiagnostic
from core.session import SessionDiagnostic, EtapeDiagnostic


def _nouvelle_session() -> tuple:
    session = SessionDiagnostic()
    agent = AgentDiagnostic()
    return session, agent


@cl.on_chat_start
async def demarrer():
    session, agent = _nouvelle_session()
    cl.user_session.set("session", session)
    cl.user_session.set("agent", agent)

    reponse = agent.traiter_message("", session)
    await cl.Message(content=reponse).send()


@cl.on_message
async def recevoir_message(message: cl.Message):
    # Commande de reset : le client tape /new ou "new report"
    if message.content.strip().lower() in ["/new", "new report", "restart", "/restart"]:
        await _reinitialiser()
        return

    session: SessionDiagnostic = cl.user_session.get("session")
    agent: AgentDiagnostic = cl.user_session.get("agent")

    # Si la session est déjà terminée, proposer de redémarrer
    if session.etape == EtapeDiagnostic.TERMINE:
        await cl.Message(
            content="This session is already complete. Type **/new** to start a new fault report."
        ).send()
        return

    async with cl.Step(name=f"Step: {session.etape.value}"):
        reponse = agent.traiter_message(message.content, session)

    await cl.Message(content=reponse).send()

    if session.etape == EtapeDiagnostic.TERMINE:
        await _afficher_recap(session)
        # Proposer un nouveau diagnostic
        await cl.Message(
            content="---\nWould you like to report another fault? Click the button below or type **/new**.",
            actions=[
                cl.Action(
                    name="new_report",
                    label="🔄 Start new report",
                    value="new",
                    description="Reset and start a new fault diagnosis"
                )
            ]
        ).send()


@cl.action_callback("new_report")
async def on_new_report(action: cl.Action):
    await _reinitialiser()


async def _reinitialiser():
    """Réinitialise la session et redémarre le diagnostic."""
    session, agent = _nouvelle_session()
    cl.user_session.set("session", session)
    cl.user_session.set("agent", agent)

    await cl.Message(content="---\n✅ New session started.").send()
    reponse = agent.traiter_message("", session)
    await cl.Message(content=reponse).send()


async def _afficher_recap(session: SessionDiagnostic):
    data = session.to_dict()
    recap = f"""
### ✅ Fault Report Recorded

| Field | Value |
|-------|-------|
| **Perimeter** | {data.get('perimeter', 'N/A')} |
| **CIF Title** | {data.get('cif_title', 'N/A')} |
| **Questions asked** | {data.get('questions_asked', 0)} |

**Summary sent to technical team:**
> {session.synthese}
"""
    await cl.Message(content=recap, author="System").send()
