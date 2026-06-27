from src.usernames import normalize_username


def set_user_password(
    db: dict,
    username: str,
    new_password: str,
    salt_factory,
    hash_password,
    updated_at: str,
) -> dict:
    normalized_username = normalize_username(username)
    if not (new_password or "").strip():
        raise ValueError("Password cannot be blank.")

    users = db.get("users", {}) if isinstance(db, dict) else {}
    if normalized_username not in users:
        raise KeyError(f"User not found: {normalized_username}")

    salt = salt_factory()
    user_record = dict(users[normalized_username])
    user_record["salt"] = salt
    user_record["hash"] = hash_password(new_password, salt)
    user_record["password_updated"] = updated_at
    users[normalized_username] = user_record
    db["users"] = users
    return db
