
from app.services.user_context import load_user_context


def build_context(
    project_context: dict,
    session_history: list[dict],
    user_key: str = "local-user",
) -> dict:
    """
    Assemble the context needed for one research request.
    """

    user_context = load_user_context(user_key)

    return {
        "user": user_context,
        "project": project_context,
        "session": {
            "recent_messages": session_history,
        },
    }
