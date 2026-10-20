
from app.database import SessionLocal
from app.models import UserProfile


DEFAULT_USER_KEY = "local-user"


def load_user_context(
    user_key: str = DEFAULT_USER_KEY,
) -> dict:
    db = SessionLocal()

    try:
        profile = (
            db.query(UserProfile)
            .filter(UserProfile.user_key == user_key)
            .first()
        )

        if profile is None:
            return {
                "preferences": {},
                "interests": [],
                "expertise": {},
            }

        return {
            "preferences": profile.preferences or {},
            "interests": profile.interests or [],
            "expertise": profile.expertise or {},
        }

    finally:
        db.close()


def update_user_context(
    updates: dict,
    user_key: str = DEFAULT_USER_KEY,
) -> dict:
    allowed_fields = {
        "preferences",
        "interests",
        "expertise",
    }

    invalid_fields = set(updates) - allowed_fields

    if invalid_fields:
        raise ValueError(
            f"Unsupported user context fields: {invalid_fields}"
        )

    db = SessionLocal()

    try:
        profile = (
            db.query(UserProfile)
            .filter(UserProfile.user_key == user_key)
            .first()
        )

        if profile is None:
            profile = UserProfile(
                user_key=user_key,
                preferences={},
                interests=[],
                expertise={},
            )
            db.add(profile)

        for field, value in updates.items():
            if field == "preferences":
                profile.preferences = {
                    **(profile.preferences or {}),
                    **value,
                }

            elif field == "expertise":
                profile.expertise = {
                    **(profile.expertise or {}),
                    **value,
                }

            elif field == "interests":
                existing = profile.interests or []
                profile.interests = list(
                    dict.fromkeys(existing + value)
                )

        db.commit()
        db.refresh(profile)

        return {
            "preferences": profile.preferences or {},
            "interests": profile.interests or [],
            "expertise": profile.expertise or {},
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
