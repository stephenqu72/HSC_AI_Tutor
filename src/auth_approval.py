from src.usernames import normalize_username


ROOT_USER = "stephenqu72@gmail.com"


def is_root_user(username: str) -> bool:
    return normalize_username(username) == ROOT_USER


def is_user_approved(username: str, user: dict | None) -> bool:
    if is_root_user(username):
        return True
    return bool(isinstance(user, dict) and user.get("approved") is True)


def llm_owner_username(username: str) -> str:
    return ROOT_USER


def can_generate_shared_answer_from_submission(username: str) -> bool:
    return bool(normalize_username(username))


def apply_approval_policy(db: dict) -> tuple[dict, bool]:
    users = db.get("users", {}) if isinstance(db, dict) else {}
    updated_users = {}
    changed = False

    for username, user in users.items():
        user_record = dict(user or {})
        if is_root_user(username):
            if user_record.get("approved") is not True:
                user_record["approved"] = True
                changed = True
            if user_record.get("role") != "root":
                user_record["role"] = "root"
                changed = True
        else:
            if "approved" not in user_record:
                user_record["approved"] = False
                changed = True
            if user_record.get("role") != "user":
                user_record["role"] = "user"
                changed = True
        updated_users[username] = user_record

    updated_db = {**(db if isinstance(db, dict) else {}), "users": updated_users}
    return updated_db, changed
