from dataclasses import dataclass
import random

from src.auth_approval import is_root_user


ROOT_GEMINI_KEY_NAMES = [
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4",
]
STUDENT_GEMINI_KEY_NAME = "GEMINI_API_KEY_stu"


@dataclass(frozen=True)
class GeminiKeySelection:
    key_name: str
    api_key: str


def select_gemini_key(username: str, key_values: dict, random_index: int | None = None) -> GeminiKeySelection:
    if is_root_user(username):
        available_keys = [
            key_name
            for key_name in ROOT_GEMINI_KEY_NAMES
            if (key_values.get(key_name) or "").strip()
        ]
        if not available_keys:
            raise ValueError(
                "No root Gemini API key is set in the environment or Streamlit secrets."
            )
        selected_index = random.randrange(len(available_keys)) if random_index is None else random_index
        key_name = available_keys[selected_index % len(available_keys)]
    else:
        key_name = STUDENT_GEMINI_KEY_NAME

    api_key = (key_values.get(key_name) or "").strip()
    if not api_key:
        raise ValueError(f"{key_name} not set in environment or Streamlit secrets.")
    return GeminiKeySelection(key_name=key_name, api_key=api_key)
