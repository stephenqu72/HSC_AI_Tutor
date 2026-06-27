import json
import os


SESSION_PREF_KEYS = {
    "practice_mode",
    "course_level",
    "selected_topic",
    "selected_subtopic",
    "selected_paper",
    "question_type_filter",
    "llm_choice",
    "selected_note_pdf",
    "question_index",
}


def load_session_prefs(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {key: data[key] for key in SESSION_PREF_KEYS if key in data}
    except Exception:
        return {}


def save_session_prefs(path: str, values: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prefs = {key: values[key] for key in SESSION_PREF_KEYS if key in values}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)
