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
    # Reset command
    if message.content.strip().lower() in ["/new", "new report", "restart", "/restart"]:
        await _reinitialiser()
        return

    session: SessionDiagnostic = cl.user_session.get("session")
    agent: AgentDiagnostic = cl.user_session.get("agent")

    if session.etape == EtapeDiagnostic.TERMINE:
        await cl.Message(
            content="This session is already complete. Type **/new** to start a new fault report."
        ).send()
        return

    async with cl.Step(name=f"Step: {session.etape.value}"):
        reponse = agent.traiter_message(message.content, session)

    await cl.Message(content=reponse).send()

    # Show "I'm done" button when entering or staying in COMPLEMENTS phase
    if session.etape == EtapeDiagnostic.COMPLEMENTS:
        await _afficher_bouton_fin()

    if session.etape == EtapeDiagnostic.TERMINE:
        await _afficher_recap(session)
        await cl.Message(
            content="Would you like to report another fault?",
            actions=[
                cl.Action(
                    name="new_report",
                    label="🔄 Start new report",
                    payload={"action": "new"}
                )
            ]
        ).send()


async def _afficher_bouton_fin():
    """Displays the persistent 'I'm done' button during the COMPLEMENTS phase."""
    await cl.Message(
        content=" ",
        actions=[
            cl.Action(
                name="finish_complements",
                label="✅ I have nothing more to add",
                payload={"action": "done"}
            )
        ]
    ).send()


@cl.action_callback("finish_complements")
async def on_finish_complements(action: cl.Action):
    """Triggered when the customer clicks 'I have nothing more to add'."""
    session: SessionDiagnostic = cl.user_session.get("session")
    agent: AgentDiagnostic = cl.user_session.get("agent")

    if session.etape != EtapeDiagnostic.COMPLEMENTS:
        return

    async with cl.Step(name="Generating final summary..."):
        # Inject a neutral closing message and call the complement handler
        reponse = agent.traiter_message("no", session)

    await cl.Message(content=reponse).send()

    if session.etape == EtapeDiagnostic.TERMINE:
        await _afficher_recap(session)
        await cl.Message(
            content="Would you like to report another fault?",
            actions=[
                cl.Action(
                    name="new_report",
                    label="🔄 Start new report",
                    payload={"action": "new"}
                )
            ]
        ).send()


@cl.action_callback("new_report")
async def on_new_report(action: cl.Action):
    await _reinitialiser()


async def _reinitialiser():
    session, agent = _nouvelle_session()
    cl.user_session.set("session", session)
    cl.user_session.set("agent", agent)

    await cl.Message(content="---\n✅ New session started.").send()
    reponse = agent.traiter_message("", session)
    await cl.Message(content=reponse).send()


async def _afficher_recap(session: SessionDiagnostic):
    data = session.to_dict()

    # Format CIF ranking
    ranking_texte = ""
    if data.get("cif_ranking"):
        lignes = ["**Most probable fault type(s):**"]
        for item in data["cif_ranking"]:
            lignes.append(
                f"- **{item.get('probabilite', '?')}%** — "
                f"[{item.get('perimeter', '')}] {item.get('cif_title', '')}"
            )
        ranking_texte = "\n".join(lignes)

    recap = f"""
### ✅ Fault Report Recorded

| Field | Value |
|-------|-------|
| **Perimeter** | {data.get('perimeter', 'N/A')} |
| **CIF Title** | {data.get('cif_title', 'N/A')} |
| **Questions asked** | {data.get('questions_asked', 0)} |

**Summary sent to technical team:**
> {session.synthese}

{ranking_texte}
"""
    await cl.Message(content=recap, author="System").send()
