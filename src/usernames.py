def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def normalize_user_db(db: dict) -> tuple[dict, bool]:
    users = db.get("users", {}) if isinstance(db, dict) else {}
    normalized_users = {}
    changed = False

    for username, user in users.items():
        normalized_username = normalize_username(username)
        if not normalized_username:
            changed = True
            continue
        if normalized_username != username:
            changed = True
        if normalized_username in normalized_users:
            changed = True
            if username == normalized_username:
                normalized_users[normalized_username] = user
            continue
        normalized_users[normalized_username] = user

    normalized_db = {**(db if isinstance(db, dict) else {}), "users": normalized_users}
    return normalized_db, changed
