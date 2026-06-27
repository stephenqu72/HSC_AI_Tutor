import streamlit as st
import os
import json
import re
from PIL import Image
from io import BytesIO
import importlib.util
import hashlib
import binascii
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import streamlit.components.v1 as components
from src.password_reset import set_user_password

try:
    from src.student_answers import (
        append_answer_log,
        build_answer_feedback_prompt,
        build_answer_summary,
        canonical_question_cache_key,
        estimate_study_panel_height,
        latest_answers_by_key,
        parse_flash_card,
        question_type_course_for_cache_key,
        read_json_list,
        study_markdown_to_html,
    )
except Exception:
    def parse_flash_card(text: str) -> dict:
        front_match = re.search(r"###\s*Front\s*([\s\S]*?)(?=###\s*Back|$)", text or "", re.IGNORECASE)
        back_match = re.search(r"###\s*Back\s*([\s\S]*)", text or "", re.IGNORECASE)
        return {
            "front": front_match.group(1).strip() if front_match else "Review this key HSC Mathematics idea.",
            "back": back_match.group(1).strip() if back_match else (text or "").strip(),
        }

    def _flash_card_inline_markdown_to_html(text: str) -> str:
        rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", text or "")
        rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
        return rendered

    def study_markdown_to_html(text: str) -> str:
        blocks = []
        list_items = []
        paragraph_lines = []

        def flush_list():
            if list_items:
                blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
                list_items.clear()

        def flush_paragraph():
            if paragraph_lines:
                blocks.append(f"<p>{_flash_card_inline_markdown_to_html(' '.join(paragraph_lines))}</p>")
                paragraph_lines.clear()

        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_list()
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                flush_paragraph()
                flush_list()
                level = min(len(heading.group(1)), 4)
                blocks.append(f"<h{level}>{_flash_card_inline_markdown_to_html(heading.group(2))}</h{level}>")
                continue
            if line.startswith(("- ", "* ")):
                flush_paragraph()
                list_items.append(_flash_card_inline_markdown_to_html(line[2:].strip()))
                continue
            flush_list()
            paragraph_lines.append(line)

        flush_paragraph()
        flush_list()
        return "\n".join(blocks)

    def estimate_study_panel_height(text: str, min_height: int = 220, max_height: int = 1800) -> int:
        raw_lines = (text or "").splitlines() or [""]
        visual_lines = 0
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                visual_lines += 1
                continue
            line_weight = 1
            if re.match(r"^#{1,6}\s+", line):
                line_weight = 2
            elif line.startswith(("- ", "* ")):
                line_weight = 1.15
            visual_lines += max(line_weight, len(line) / 82)
        estimated = 118 + int(visual_lines * 29)
        return max(min_height, min(max_height, estimated))
from src.usernames import normalize_user_db, normalize_username
from src.auth_approval import apply_approval_policy, is_root_user, is_user_approved, llm_owner_username
from src.gemini_keys import ROOT_GEMINI_KEY_NAMES, STUDENT_GEMINI_KEY_NAME, select_gemini_key
from src.session_prefs import load_session_prefs, save_session_prefs

############################################
# 💼 Multi-user Auth + Per-user Storage (Streamlit Cloud ready)
############################################
APP_TITLE = "HSC AI Tutoring Centre"

# Use secrets or env to configure base paths for deployment
PERSIST_DIR = os.getenv("PERSIST_DIR") or st.secrets.get("PERSIST_DIR", "persist")
BASE_ROOT = os.getenv("BASE_ROOT") or st.secrets.get("BASE_ROOT", "data/HSCMath")
NOTES_ROOT = os.getenv("NOTES_ROOT") or st.secrets.get("NOTES_ROOT", os.path.join(os.path.dirname(BASE_ROOT), "Notes"))

ACCOUNTS_DB = os.path.join(PERSIST_DIR, "server", "users.json")
USERS_ROOT = os.path.join(PERSIST_DIR, "users")

# ensure required dirs
os.makedirs(os.path.dirname(ACCOUNTS_DB), exist_ok=True)
os.makedirs(USERS_ROOT, exist_ok=True)

# password hashing helpers (no external deps)
_DEF_ITER = 200_000
_DEF_ALG = "sha256"


def _new_salt() -> str:
    return binascii.hexlify(os.urandom(16)).decode()


def _hash_pw(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(_DEF_ALG, password.encode("utf-8"), binascii.unhexlify(salt), _DEF_ITER)
    return binascii.hexlify(dk).decode()


def load_users() -> dict:
    if not os.path.exists(ACCOUNTS_DB):
        os.makedirs(os.path.dirname(ACCOUNTS_DB), exist_ok=True)
        with open(ACCOUNTS_DB, "w", encoding="utf-8") as f:
            json.dump({"users": {}}, f)
        return {"users": {}}
    with open(ACCOUNTS_DB, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(db: dict):
    os.makedirs(os.path.dirname(ACCOUNTS_DB), exist_ok=True)
    with open(ACCOUNTS_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def ensure_user_space(username: str):
    root = os.path.join(USERS_ROOT, username)
    fb_dir = os.path.join(root, "feedback")
    tmp_dir = os.path.join(root, "temp")
    os.makedirs(fb_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    return root, fb_dir, tmp_dir


def load_auth_db() -> dict:
    db = load_users()
    db, usernames_changed = normalize_user_db(db)
    db, approval_changed = apply_approval_policy(db)
    if usernames_changed or approval_changed:
        save_users(db)
    return db


def update_user_password(db: dict, username: str, new_password: str) -> dict:
    return set_user_password(
        db,
        username,
        new_password,
        salt_factory=_new_salt,
        hash_password=_hash_pw,
        updated_at=datetime.utcnow().isoformat() + "Z",
    )


############################################
# 🌟 App Settings
############################################
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --------- Auth UI ---------
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
elif st.session_state.auth_user is not None:
    st.session_state.auth_user = normalize_username(st.session_state.auth_user)

st.markdown("<h1 style='text-align: center;'> Your personal HSC Assistant ✨</h1>", unsafe_allow_html=True)

if st.session_state.auth_user is None:
    colA, colB = st.columns([2, 1])
    with colA:
        username = st.text_input("Username (e.g., email)", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
    with colB:
        st.caption("Tip: If this username doesn't exist, we'll create it for you.")
        go = st.button("Sign in / Create Account", use_container_width=True)

    if go:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            username = normalize_username(username)
            db = load_users()
            db, usernames_changed = normalize_user_db(db)
            db, approval_changed = apply_approval_policy(db)
            if usernames_changed or approval_changed:
                save_users(db)

            user = db["users"].get(username)
            if user is None:
                salt = _new_salt()
                pw_hash = _hash_pw(password, salt)
                approved = is_root_user(username)
                db["users"][username] = {
                    "salt": salt,
                    "hash": pw_hash,
                    "created": datetime.utcnow().isoformat() + "Z",
                    "approved": approved,
                    "role": "root" if approved else "user",
                }
                save_users(db)
                ensure_user_space(username)
                if approved:
                    st.success("Account created and signed in ✨")
                    st.session_state.auth_user = username
                    st.rerun()
                else:
                    st.info("Account created and waiting for approval by stephenqu72@gmail.com.")
            else:
                calc = _hash_pw(password, user["salt"])
                if calc == user["hash"]:
                    if is_user_approved(username, user):
                        ensure_user_space(username)
                        st.success("Signed in ✅")
                        st.session_state.auth_user = username
                        st.rerun()
                    else:
                        st.warning("Your account is waiting for approval by stephenqu72@gmail.com.")
                else:
                    st.error("Incorrect password. Please try again.")
db = load_auth_db()
active_user = db.get("users", {}).get(st.session_state.auth_user) if st.session_state.auth_user else None
if st.session_state.auth_user is None:
    st.stop()
if not is_user_approved(st.session_state.auth_user, active_user):
    st.warning("Your account is waiting for approval by stephenqu72@gmail.com.")
    st.session_state.auth_user = None
    st.stop()

current_user = normalize_username(st.session_state.auth_user)
st.session_state.auth_user = current_user
user_root, user_fb_dir, user_tmp_dir = ensure_user_space(current_user)
USER_SESSION_PREFS_FILE = os.path.join(user_root, "ui_session_state.json")
if st.session_state.get("_session_prefs_loaded_for") != current_user:
    for key, value in load_session_prefs(USER_SESSION_PREFS_FILE).items():
        st.session_state[key] = value
    st.session_state["_session_prefs_loaded_for"] = current_user

st.markdown(
    f"<p style='text-align: center;'>Be a star today, {current_user}! ⭐ Your data is saved in <code>{user_root}</code></p>",
    unsafe_allow_html=True,
)


def render_account_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.success(f"Signed in as **{current_user}**")
    if st.sidebar.button("Sign out"):
        st.session_state.auth_user = None
        st.rerun()

    with st.sidebar.expander("🔑 Change Password", expanded=False):
        current_password = st.text_input("Current password", type="password", key="self_current_password")
        new_password = st.text_input("New password", type="password", key="self_new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="self_confirm_password")
        if st.button("Update My Password", key="self_update_password"):
            if not current_password or not new_password or not confirm_password:
                st.warning("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.warning("New passwords do not match.")
            else:
                auth_db = load_auth_db()
                user = auth_db.get("users", {}).get(current_user)
                if not user or _hash_pw(current_password, user["salt"]) != user["hash"]:
                    st.error("Current password is incorrect.")
                else:
                    try:
                        auth_db = update_user_password(auth_db, current_user, new_password)
                        save_users(auth_db)
                        st.success("Password updated.")
                    except Exception as e:
                        st.error(f"Unable to update password: {e}")

############################################
# 🧠 Gemini API Setup
############################################
load_dotenv()


def get_config_value(name: str):
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


gemini_key_values = {
    key_name: get_config_value(key_name)
    for key_name in [*ROOT_GEMINI_KEY_NAMES, STUDENT_GEMINI_KEY_NAME]
}


def configure_gemini_for_current_user():
    gemini_key_selection = select_gemini_key(current_user, gemini_key_values)
    genai.configure(api_key=gemini_key_selection.api_key)
    return gemini_key_selection


try:
    configure_gemini_for_current_user()
except ValueError as e:
    st.error(str(e))
    st.stop()


def render_root_admin_sidebar():
    if not is_root_user(current_user):
        return

    auth_db = load_auth_db()
    users = auth_db.get("users", {})
    managed_users = sorted(username for username in users if not is_root_user(username))

    st.sidebar.markdown("## 🔐 User Approval")
    if not managed_users:
        st.sidebar.caption("No student accounts yet.")
        return

    pending_users = [username for username in managed_users if not users[username].get("approved")]
    approved_users = [username for username in managed_users if users[username].get("approved")]

    with st.sidebar.expander("Pending accounts", expanded=True):
        if not pending_users:
            st.caption("No pending accounts.")
        for username in pending_users:
            st.caption(username)
            if st.button("Approve", key=f"approve_{hashlib.sha1(username.encode('utf-8')).hexdigest()}"):
                users[username]["approved"] = True
                users[username]["approved_by"] = current_user
                users[username]["approved_at"] = datetime.utcnow().isoformat() + "Z"
                save_users(auth_db)
                st.rerun()

        with st.sidebar.expander("Approved accounts", expanded=False):
            if not approved_users:
                st.caption("No approved student accounts.")
            for username in approved_users:
                st.caption(username)
                if st.button("Revoke", key=f"revoke_{hashlib.sha1(username.encode('utf-8')).hexdigest()}"):
                    users[username]["approved"] = False
                    users[username]["revoked_by"] = current_user
                    users[username]["revoked_at"] = datetime.utcnow().isoformat() + "Z"
                    save_users(auth_db)
                    st.rerun()

        reset_options = sorted(users)
        with st.sidebar.expander("Reset User Password", expanded=False):
            if not reset_options:
                st.caption("No accounts to reset.")
            else:
                reset_user = st.selectbox("User", reset_options, key="root_reset_user")
                root_new_password = st.text_input("New password", type="password", key="root_reset_new_password")
                root_confirm_password = st.text_input("Confirm password", type="password", key="root_reset_confirm_password")
                if st.button("Reset Password", key="root_reset_password"):
                    if not root_new_password or not root_confirm_password:
                        st.warning("Please enter and confirm the new password.")
                    elif root_new_password != root_confirm_password:
                        st.warning("Passwords do not match.")
                    else:
                        try:
                            auth_db = update_user_password(auth_db, reset_user, root_new_password)
                            users[reset_user]["password_reset_by"] = current_user
                            users[reset_user]["password_reset_at"] = datetime.utcnow().isoformat() + "Z"
                            save_users(auth_db)
                            st.success(f"Password reset for {reset_user}.")
                        except Exception as e:
                            st.error(f"Unable to reset password: {e}")


############################################
# 🎯 Prompt Template
############################################
prompt_generating = """
You are a top HSC teacher. Based on the concepts shown in the input image, generate a high-quality Y12 NSW HSC-style question (not limited to multiple choice). Follow this structure:
{
  "Question": "The question in text format, including mathematical notation if applicable.",
  "Question_Type": "one of: 'OpenEnded', 'Proof', 'MultipleChoice', 'ShortAnswer', 'TableCompletion', etc.",
  "Image_DataTable": "Python code to generate any diagram or data table, using Plotly or Matplotlib. Must define a function `generate_plot()` that returns a figure.",
  "Multiple Choices": ["A", "B", "C", "D"],
  "Answer": "Correct answer or marking guide.",
  "Marks": "Total marks allocated, if applicable."
}
⚠️ IMPORTANT: Please format all LaTeX math expressions using:
- `$...$` for inline math
- `$$...$$` for block math

Do not use `\\(...\\)` or `\\[...\\]`.
"""

############################################
# ---------- Helpers ----------
############################################

def extract_json_from_response(response_text):
    raw_text = response_text.text.strip()
    match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_text)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"(\{[\s\S]+\})", raw_text)
        if match:
            json_str = match.group(1)
        else:
            raise ValueError("⚠️ No JSON object found in Gemini response.")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"⚠️ JSON decoding failed: {e}\n\nContent was:\n{json_str[:300]}...")


def load_plot_module(module_path: str):
    spec = importlib.util.spec_from_file_location("image_module", module_path)
    image_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_module)
    return image_module


QUESTION_TYPES = [
    "Multiple choice",
    "Short answer",
    "Calculation question",
    "Essay",
    "Experimental questions",
    "Data analysis questions",
    "Other questions",
]
QUESTION_TYPE_FILTERS = ["All types"] + QUESTION_TYPES
DEFAULT_QUESTION_TYPE = "Other questions"


def normalize_question_type(value: str) -> str:
    cleaned = (value or "").strip().lower()
    aliases = {
        "multiple choice": "Multiple choice",
        "multiple-choice": "Multiple choice",
        "mcq": "Multiple choice",
        "short answer": "Short answer",
        "short-answer": "Short answer",
        "calculation": "Calculation question",
        "calculation question": "Calculation question",
        "numerical": "Calculation question",
        "essay": "Essay",
        "extended response": "Essay",
        "experimental": "Experimental questions",
        "experiment": "Experimental questions",
        "experimental question": "Experimental questions",
        "experimental questions": "Experimental questions",
        "data analysis": "Data analysis questions",
        "data analysis question": "Data analysis questions",
        "data analysis questions": "Data analysis questions",
        "other": "Other questions",
        "other question": "Other questions",
        "other questions": "Other questions",
    }
    return aliases.get(cleaned, DEFAULT_QUESTION_TYPE)


def read_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_question_type(path: str, key: str) -> str:
    data = read_json_dict(path)
    entry = data.get(key, {})
    if isinstance(entry, dict):
        return normalize_question_type(entry.get("type"))
    return normalize_question_type(entry)


def question_type_exists(path: str, key: str) -> bool:
    return key in read_json_dict(path)


def save_question_type(path: str, key: str, question_type: str, metadata: dict):
    db = read_json_dict(path)
    db[key] = {
        "type": normalize_question_type(question_type),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        **(metadata or {}),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def classify_question_type_from_text(text: str) -> str:
    marker = re.search(r"###QUESTION_TYPE\s*([\s\S]+?)$", text or "", re.IGNORECASE)
    if marker:
        return normalize_question_type(marker.group(1).strip().splitlines()[0])
    return DEFAULT_QUESTION_TYPE


def strip_question_type_section(text: str) -> str:
    if not text:
        return ""
    return re.split(r"###QUESTION_TYPE[\s\S]*$", text, flags=re.IGNORECASE)[0].strip()


def list_note_pdfs(notes_root: str) -> list:
    if not os.path.isdir(notes_root):
        return []
    pdfs = []
    for root, _, files in os.walk(notes_root):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                path = os.path.join(root, filename)
                pdfs.append(os.path.relpath(path, notes_root))
    return sorted(pdfs, key=str.lower)


@st.cache_data(show_spinner=False)
def pdf_page_count(pdf_path: str) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


@st.cache_data(show_spinner=False)
def render_pdf_page(pdf_path: str, page_number: int, zoom: float = 1.8) -> bytes:
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def show_pdf_page_viewer(pdf_path: str, viewer_key: str):
    try:
        total_pages = pdf_page_count(pdf_path)
    except ImportError:
        st.error("PyMuPDF is required to render PDF notes. Install the PyMuPDF package.")
        return
    except Exception as e:
        st.warning(f"⚠️ Unable to open this PDF: {e}")
        return

    if total_pages <= 0:
        st.warning("⚠️ This PDF has no pages to show.")
        return

    page_key = f"note_page_{viewer_key}"
    input_key = f"note_page_input_{viewer_key}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = min(max(st.session_state[page_key], 1), total_pages)
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[page_key]
    st.session_state[input_key] = min(max(st.session_state[input_key], 1), total_pages)

    def set_note_page(delta=0):
        next_page = min(max(st.session_state[page_key] + delta, 1), total_pages)
        st.session_state[page_key] = next_page
        st.session_state[input_key] = next_page

    def sync_note_page_input():
        st.session_state[page_key] = min(max(st.session_state[input_key], 1), total_pages)

    nav_prev, nav_page, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        st.button(
            "⬅️ Previous page",
            key=f"prev_{viewer_key}",
            disabled=st.session_state[page_key] <= 1,
            on_click=set_note_page,
            args=(-1,),
        )
    with nav_page:
        st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key=input_key,
            on_change=sync_note_page_input,
            label_visibility="collapsed",
        )
        st.caption(f"Page {st.session_state[page_key]} of {total_pages}")
    with nav_next:
        st.button(
            "Next page ➡️",
            key=f"next_{viewer_key}",
            disabled=st.session_state[page_key] >= total_pages,
            on_click=set_note_page,
            args=(1,),
        )

    try:
        page_png = render_pdf_page(pdf_path, st.session_state[page_key])
        st.image(page_png, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Unable to render this PDF page: {e}")


def read_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # migrate older dict-style file to list with single entry
            return [data]
        else:
            return []
    except Exception:
        return []


def append_json_log(path: str, entry: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = read_json_list(path)
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_last_feedback_for_key(path: str, key: str):
    """Return the most recent feedback entry for a given logical key (or None)."""
    data = read_json_list(path)
    for item in reversed(data):
        if item.get("key") == key:
            return item
    return None

def get_all_feedback_for_key(path: str, key: str):
    """Return all feedback entries for a key, sorted ascending by timestamp."""
    data = [d for d in read_json_list(path) if d.get("key") == key]
    def _parse_ts(x):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            return datetime.min
    data.sort(key=lambda d: _parse_ts(d.get("ts", "")))
    return data


def extract_dot(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:dot|graphviz)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(digraph\s+[A-Za-z0-9_]*\s*\{[\s\S]*?\})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""

def image_to_b64_png(pil_img: Image.Image) -> str:
    import base64
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_image_for_model(img_path: str):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
    return Image.open(BytesIO(img_bytes))

# Sort by numeric suffix before .png

def extract_picture_number(filename):
    match = re.search(r'(?:Picture|Group)\s(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0

def call_model(prompt, image):
    #model = genai.GenerativeModel("gemini-2.5-flash")
    model = genai.GenerativeModel(selected_model)
    return model.generate_content([prompt, image])

def _extract_plot_code(data: dict) -> str:
      b64 = (data.get("Image_DataTable_b64") or "").strip()
      if b64:
          try:
              return base64.b64decode(b64).decode("utf-8", errors="replace")
          except Exception:
              pass  # fall back to plain
      return (data.get("Image_DataTable") or "").strip()

def extract_json_from_response_tolerant(response_text):
    """
    Parse a JSON object from an LLM response, tolerating bad escaping inside
    Image_DataTable. Strategy:
      1) Try the strict extractor.
      2) If that fails, find a JSON-looking block (```json ...``` or { ... }).
      3) Try json.loads; if it fails, remove/empty Image_DataTable and reparse.
      4) If a separate fenced Python code block exists, attach it back as Image_DataTable.
    Returns: dict
    """
    raw = getattr(response_text, "text", str(response_text)).strip()

    # 1) Try your strict path first
    try:
        return extract_json_from_response(response_text)
    except Exception:
        pass  # continue with tolerant path

    # 2) Find a JSON candidate (prefer fenced json)
    m_fenced_json = re.search(r"```json\s*([\s\S]+?)\s*```", raw, re.IGNORECASE)
    if m_fenced_json:
        json_candidate = m_fenced_json.group(1)
    else:
        m_obj = re.search(r"(\{[\s\S]+?\})", raw)
        if not m_obj:
            raise ValueError("⚠️ Couldn't locate a JSON object in the model response.")
        json_candidate = m_obj.group(1)

    # 3) Try plain parse
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass

    # 3a) Capture a separate fenced python block (if any) to reattach later
    code_in_fence = ""
    # Prefer a python fence that is NOT the json fence we already used
    for lang in ("python", "py"):
        m_code = re.search(rf"```{lang}\s*([\s\S]+?)\s*```", raw, re.IGNORECASE)
        if m_code:
            code_in_fence = m_code.group(1)
            break
    if not code_in_fence:
        # fallback: any non-json fenced block
        m_any = re.search(r"```(?!json)[A-Za-z0-9_+-]*\s*([\s\S]+?)\s*```", raw, re.IGNORECASE)
        if m_any:
            code_in_fence = m_any.group(1)

    # 4) Try to blank out or remove the Image_DataTable value and reparse
    #    (first attempt: replace its value with empty string)
    pat_value = r'"Image_DataTable"\s*:\s*"[\s\S]*?"'
    json_fixed = re.sub(pat_value, '"Image_DataTable": ""', json_candidate)
    # also clean trailing commas like ,}
    json_fixed = re.sub(r",\s*}", "}", json_fixed)
    try:
        data = json.loads(json_fixed)
    except json.JSONDecodeError:
        # second attempt: remove the whole field (plus trailing comma if present)
        pat_field = r'"Image_DataTable"\s*:\s*"[\s\S]*?"\s*,?'
        json_fixed2 = re.sub(pat_field, "", json_candidate)
        json_fixed2 = re.sub(r",\s*}", "}", json_fixed2)
        data = json.loads(json_fixed2)

    # 5) Reattach code if we captured any
    if code_in_fence:
        data["Image_DataTable"] = code_in_fence
    else:
        data.setdefault("Image_DataTable", "")

    return data


def call_text_model(prompt):
    model = genai.GenerativeModel(selected_model)
    return model.generate_content(prompt)


def build_flash_card_prompt(saved_answer: str) -> str:
    return f"""
You are a concise NSW HSC Mathematics study coach. Create one flash card from the saved answer below.

Saved answer:
{saved_answer}

Format exactly:
### Front
A short recall question about the key formula, theorem, method, definition, graph feature, or common trap.

### Back
- Key idea:
- Formula / theorem / method:
- When to use it:
- Common trap:

Keep it compact and exam-focused. If no fixed formula or theorem applies, write "No fixed formula or theorem".
""".strip()


TEXT_ANSWER_PROMPT = """
You are a top NSW HSC Mathematics teacher. Read the image and answer the question carefully.

Return plain text only using exactly these sections:

###ANSWER
(concise final answer)

###EXPLANATION
(step-by-step reasoning)

###OTHERS
(optional extra tips, traps, or shortcuts)

If the question needs a diagram or graph, mention it naturally in the answer, but do not return JSON.

⚠️ IMPORTANT: Use `$...$` for inline math and `$$...$$` for display math.
Do not use `\\(...\\)` or `\\[...\\]`.
""".strip()


GRAPH_ANSWER_PROMPT = """
You are a top NSW HSC Mathematics teacher. Read the image and answer the question carefully.

Return plain text only using exactly these sections:

###ANSWER
(concise final answer)

###PLOT_CODE
If a graph, diagram, or plot would help, provide Python code that defines generate_plot() and returns a Plotly or Matplotlib figure as fig. If no plot is needed, leave this blank.

###EXPLANATION
(step-by-step reasoning)

###OTHERS
(optional extra tips, traps, or shortcuts)

⚠️ IMPORTANT: Use `$...$` for inline math and `$$...$$` for display math.
Do not use `\\(...\\)` or `\\[...\\]`.
""".strip()


def display_text_answer(reply: str, question_type: str):
    st.markdown("### ✅ Answer")
    st.caption(f"Question type: {question_type}")
    st.markdown(strip_question_type_section(reply))


def display_graph_answer(raw: str, base_no_ext: str):
    data = {"ANSWER": "", "PLOT_CODE": "", "EXPLANATION": "", "OTHERS": ""}

    def extract_section(text, key):
        marker = f"###{key}"
        if marker not in text:
            return ""
        part = text.split(marker, 1)[1]
        for next_header in ["###ANSWER", "###PLOT_CODE", "###EXPLANATION", "###OTHERS"]:
            if next_header != marker and next_header in part:
                part = part.split(next_header, 1)[0]
        return part.strip()

    data["ANSWER"] = extract_section(raw, "ANSWER")
    data["PLOT_CODE"] = extract_section(raw, "PLOT_CODE")
    data["EXPLANATION"] = extract_section(raw, "EXPLANATION")
    data["OTHERS"] = extract_section(raw, "OTHERS")

    if data["ANSWER"]:
        st.markdown("### ✅ Answer")
        st.markdown(data["ANSWER"])

    if data["PLOT_CODE"]:
        try:
            explain_py = os.path.join(user_tmp_dir, f"explain_plot_{base_no_ext}.py")
            with open(explain_py, "w", encoding="utf-8") as f:
                f.write(data["PLOT_CODE"])
            mod = load_plot_module(explain_py)
            fig = getattr(mod, "generate_plot", lambda: None)()
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Unable to execute plot code: {e}")

    if data["EXPLANATION"]:
        st.markdown("### 📘 Detailed Explanation")
        st.markdown(data["EXPLANATION"])

    if data["OTHERS"]:
        st.markdown("### 🧩 Other Information")
        st.markdown(data["OTHERS"])


def render_text_answer_card(reply: str, question_type: str, card_key: str):
    answer_text = strip_question_type_section(reply)
    answer_html = study_markdown_to_html(answer_text)
    panel_height = estimate_study_panel_height(answer_text)
    element_id = f"answer-card-{hashlib.sha1(card_key.encode('utf-8')).hexdigest()}"
    components.html(
        f"""
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    }},
    svg: {{ fontCache: 'global' }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  #{element_id} {{
    box-sizing: border-box;
    width: 100%;
    padding: 22px 24px;
    border: 1px solid #d9dee8;
    border-radius: 8px;
    background: linear-gradient(135deg, #f8fbff, #fff8ed);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    color: #172033;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #{element_id} .answer-card-label {{
    display: inline-block;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #5b6475;
    text-transform: uppercase;
  }}
  #{element_id} .answer-card-type {{
    margin: 0 0 16px;
    color: #687386;
    font-size: 13px;
  }}
  #{element_id} .answer-card-content {{
    font-size: 16px;
    line-height: 1.58;
  }}
  #{element_id} p {{
    margin: 0 0 12px;
  }}
  #{element_id} ul {{
    margin: 0 0 12px;
    padding-left: 22px;
  }}
  #{element_id} li {{
    margin: 0 0 9px;
  }}
  #{element_id} h1,
  #{element_id} h2,
  #{element_id} h3,
  #{element_id} h4 {{
    margin: 16px 0 10px;
    color: #172033;
    font-size: 18px;
    line-height: 1.25;
  }}
  #{element_id} h1:first-child,
  #{element_id} h2:first-child,
  #{element_id} h3:first-child,
  #{element_id} h4:first-child {{
    margin-top: 0;
  }}
  #{element_id} code {{
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.78);
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 0.95em;
  }}
  #{element_id} strong {{
    font-weight: 750;
  }}
</style>
<section id="{element_id}">
  <div class="answer-card-label">Answer</div>
  <div class="answer-card-type">Question type: {html.escape(question_type or DEFAULT_QUESTION_TYPE)}</div>
  <div class="answer-card-content">{answer_html}</div>
</section>
""",
        height=panel_height,
    )


def generate_text_answer_for_question(
    img_path: str,
    cache_key: str,
    question_type_file: str,
    metadata: dict,
    force_regen: bool = False,
):
    reply = None if force_regen else load_saved_answer(cache_key, "text")
    if not (reply or "").strip():
        image = load_image_for_model(img_path)
        response = call_model(TEXT_ANSWER_PROMPT, image)
        reply = response.text
        if not (reply or "").strip():
            raise RuntimeError("LLM returned an empty Text Answer.")
        save_answer(cache_key, "text", reply)

    question_type = classify_question_type_from_text(reply)
    save_question_type(question_type_file, cache_key, question_type, metadata)
    return reply, question_type


def answer_cache_path(cache_key: str, answer_type: str) -> str:
    cache_root = os.path.join(user_root, "saved_answers")
    os.makedirs(cache_root, exist_ok=True)
    cache_name = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    return os.path.join(cache_root, f"{cache_name}.{answer_type}.txt")


def load_saved_answer(cache_key: str, answer_type: str):
    path = answer_cache_path(cache_key, answer_type)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def save_answer(cache_key: str, answer_type: str, text: str) -> str:
    path = answer_cache_path(cache_key, answer_type)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")
    return path


def render_flash_card(card_text: str, card_key: str):
    card = parse_flash_card(card_text)
    front_html = study_markdown_to_html(card["front"])
    back_html = study_markdown_to_html(card["back"])
    element_id = f"flash-card-{hashlib.sha1(card_key.encode('utf-8')).hexdigest()}"
    components.html(
        f"""
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    }},
    svg: {{ fontCache: 'global' }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  .flash-card-wrap {{
    width: 100%;
    min-height: 280px;
    perspective: 1200px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .flash-card-toggle {{
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }}
  .flash-card {{
    display: block;
    width: 100%;
    min-height: 260px;
    cursor: pointer;
  }}
  .flash-card-inner {{
    position: relative;
    width: 100%;
    min-height: 260px;
    transition: transform 0.5s ease;
    transform-style: preserve-3d;
  }}
  .flash-card-toggle:checked + .flash-card .flash-card-inner {{
    transform: rotateY(180deg);
  }}
  .flash-card-face {{
    position: absolute;
    inset: 0;
    box-sizing: border-box;
    padding: 22px 24px;
    border: 1px solid #d9dee8;
    border-radius: 8px;
    backface-visibility: hidden;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
    color: #172033;
    overflow: auto;
  }}
  .flash-card-front {{
    background: linear-gradient(135deg, #fff7d6, #e8f4ff);
  }}
  .flash-card-back {{
    background: linear-gradient(135deg, #e9fff3, #fff1f1);
    transform: rotateY(180deg);
  }}
  .flash-card-label {{
    display: inline-block;
    margin-bottom: 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #5b6475;
    text-transform: uppercase;
  }}
  .flash-card-content {{
    font-size: 17px;
    line-height: 1.55;
    white-space: normal;
  }}
  .flash-card-content p {{
    margin: 0 0 12px;
  }}
  .flash-card-content ul {{
    margin: 0;
    padding-left: 22px;
  }}
  .flash-card-content li {{
    margin: 0 0 10px;
  }}
  .flash-card-content code {{
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.72);
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 0.95em;
  }}
  .flash-card-content strong {{
    font-weight: 750;
  }}
</style>
<div class="flash-card-wrap">
  <input class="flash-card-toggle" id="{element_id}" type="checkbox">
  <label class="flash-card" for="{element_id}">
    <div class="flash-card-inner">
      <section class="flash-card-face flash-card-front">
        <div class="flash-card-label">Front</div>
        <div class="flash-card-content">{front_html}</div>
      </section>
      <section class="flash-card-face flash-card-back">
        <div class="flash-card-label">Back</div>
        <div class="flash-card-content">{back_html}</div>
      </section>
    </div>
  </label>
</div>
""",
        height=300,
    )


############################################
# ---------- Session Init ----------
############################################
_defaults = {
    "folder": "",
    "image_files": [],
    "question_index": 0,
    "questions": {},
    "user_answers": {},
    "submitted": False,
    "last_selected_course": None,
    "last_selected_topic": None,
    "last_selected_subtopic": None,
    "feedback_cache": {},
    "thinking_map_dot": "",
    "last_mode": None,
    "last_paper": None,
    "selected_paper": "",
    "gemini_chat_history": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


############################################
# ---------- Sidebar: Course & Mode Selection ----------
############################################
st.sidebar.markdown("## 🔍 Select HSC Course")
course_level = st.sidebar.selectbox("🎓 Course:", ["Math_4U", "Math_3U", "Math_2U"], key="course_level")

st.sidebar.markdown("## 🧭 Select Practice Mode")
mode = st.sidebar.radio("Choose practice mode:", ["Topic by Topic", "Past Paper"], key="practice_mode")

base_root = os.path.join(BASE_ROOT, course_level)
if not os.path.isdir(base_root):
    st.warning(f"⚠️ The path '{base_root}' was not found. Please check your course folder structure.")
    st.stop()

selected_question_type = 'All types'
render_account_sidebar()
render_root_admin_sidebar()

############################################
# ---------- Topic or Past Paper Selection ----------
############################################
past_paper_images = []
if mode == "Past Paper":
    list_file_path = os.path.join(BASE_ROOT, f"{course_level}_List.txt")
    if not os.path.exists(list_file_path):
        st.error(f"❌ Past paper list file '{list_file_path}' not found in BASE_ROOT.")
        st.stop()

    with open(list_file_path, "r", encoding="utf-8") as f:
        paper_names = [line.strip() for line in f if line.strip()]

    selected_paper = st.sidebar.selectbox("📄 Select Past Paper:", paper_names)
    st.session_state.selected_paper = selected_paper

    question_library = os.path.join(BASE_ROOT, course_level)
    if not os.path.exists(question_library):
        st.error(f"❌ Question library folder not found at: {question_library}")
        st.stop()

    paper_prefix = selected_paper
    past_paper_images = []
    for root, _, files in os.walk(question_library):
        for f in files:
            if f.startswith(paper_prefix) and f.endswith(".png"):
                past_paper_images.append(os.path.relpath(os.path.join(root, f), question_library))

    past_paper_images.sort(key=extract_picture_number)

    if not past_paper_images:
        st.warning(f"⚠️ No questions found for '{selected_paper}' in {question_library}.")
        st.stop()

    st.session_state.folder = question_library
    st.session_state.image_files = past_paper_images

else:
    st.sidebar.markdown("## 🔍 Select Your Topic")
    topics = [f for f in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, f))]
    if not topics:
        st.warning(f"⚠️ No topics found under {base_root}.")
        st.stop()

    selected_topic = st.sidebar.selectbox("📘 Main Topic:", topics)

    subtopic_path = os.path.join(base_root, selected_topic)
    subtopics = [f for f in os.listdir(subtopic_path) if os.path.isdir(os.path.join(subtopic_path, f))]
    if not subtopics:
        st.warning(f"⚠️ No sub-topics found under {subtopic_path}.")
        st.stop()

    selected_subtopic = st.sidebar.selectbox("📁 Sub-topic:", subtopics)
    folder_path = os.path.join(subtopic_path, selected_subtopic)

    if st.session_state.folder != folder_path:
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])
        st.session_state.folder = folder_path
        st.session_state.image_files = image_files

# 🔄 LLM Model Selection
st.sidebar.markdown("## 🧠 Choose LLM Model")
selected_model = st.sidebar.selectbox("LLM Provider:", ["gemini-3.1-flash-lite","gemini-3.5-flash"], key="llm_choice")

question_type_file = os.path.join(user_root, f"question_type_{course_level}.json")
if st.sidebar.checkbox("Filter by question type", value=False, key="enable_question_type_filter"):
    if st.session_state.get("question_type_filter") not in QUESTION_TYPE_FILTERS:
        st.session_state.question_type_filter = "All types"
    selected_question_type = st.sidebar.selectbox("🏷️ Type of question:", QUESTION_TYPE_FILTERS, key="question_type_filter")
else:
    selected_question_type = "All types"

st.sidebar.markdown("## 📚 Notes")
note_pdf_options = []
if os.path.isdir(NOTES_ROOT):
    try:
        note_pdf_options = list_note_pdfs(NOTES_ROOT)
    except Exception:
        note_pdf_options = []
selected_note_pdf = None
if note_pdf_options:
    note_options = ["None"] + note_pdf_options
    if st.session_state.get("selected_note_pdf") not in note_options:
        st.session_state.selected_note_pdf = "None"
    note_choice = st.sidebar.selectbox("Open note PDF:", note_options, key="selected_note_pdf")
    selected_note_pdf = note_choice if note_choice != "None" else None
else:
    st.sidebar.caption("No PDF notes available yet.")

# Determine full image path depending on mode
if mode == "Past Paper":
    folder_path = st.session_state.folder
    current_topic_key = "PastPaper"
    current_subtopic_key = st.session_state.selected_paper.replace(" ", "_")
else:
    selected_topic = selected_topic if 'selected_topic' in locals() else st.session_state.last_selected_topic
    selected_subtopic = selected_subtopic if 'selected_subtopic' in locals() else st.session_state.last_selected_subtopic
    folder_path = os.path.join(base_root, selected_topic, selected_subtopic)
    current_topic_key = selected_topic
    current_subtopic_key = selected_subtopic

current_course_key = course_level
current_paper_key = st.session_state.get("selected_paper", "") if mode == "Past Paper" else ""
current_question_type_key = selected_question_type if mode == "Topic by Topic" else "All types"

############################################
# ---------- Selection Changed ----------
############################################
selection_changed = (
    st.session_state.last_selected_course != current_course_key or
    st.session_state.last_selected_topic != current_topic_key or
    st.session_state.last_selected_subtopic != current_subtopic_key or
    st.session_state.last_selected_question_type != current_question_type_key or
    st.session_state.last_mode != mode or
    st.session_state.last_paper != current_paper_key
)

if selection_changed:
    st.session_state.question_index = 0
    st.session_state.questions = {}
    st.session_state.user_answers = {}
    st.session_state.gemini_chat_history = []
    st.session_state.last_selected_course = current_course_key
    st.session_state.last_selected_topic = current_topic_key
    st.session_state.last_selected_subtopic = current_subtopic_key
    st.session_state.last_selected_question_type = current_question_type_key
    st.session_state.last_mode = mode
    st.session_state.last_paper = current_paper_key
    st.session_state.thinking_map_dot = ""

############################################
# ---------- Per-user, per-course feedback LOG (append-only) ----------
############################################
FEEDBACK_FILE = os.path.join(user_fb_dir, f"question_feedback_{course_level}.json")
st.sidebar.caption(f"🗂️ Feedback log (per-user, append-only): **{FEEDBACK_FILE}**")

save_session_prefs(
    USER_SESSION_PREFS_FILE,
    {
        "practice_mode": mode,
        "course_level": course_level,
        "selected_topic": current_topic_key,
        "selected_subtopic": current_subtopic_key,
        "selected_paper": st.session_state.get("selected_paper", ""),
        "question_type_filter": selected_question_type,
        "llm_choice": selected_model,
        "question_index": st.session_state.get("question_index", 0),
    },
)

USER_ANSWER_FILE = os.path.join(user_fb_dir, f"user_answers_{course_level}.json")

if mode == "Topic by Topic" and selected_question_type != "All types" and os.path.exists(question_type_file):
    filtered_images = []
    for image_name in st.session_state.image_files:
        filter_key = question_cache_key(course_level, current_topic_key, current_subtopic_key, image_name)
        if get_question_type(question_type_file, filter_key) == selected_question_type:
            filtered_images.append(image_name)
    if filtered_images:
        st.session_state.image_files = filtered_images

show_answer_summary = st.sidebar.button("📊 Show Answer Summary")

st.sidebar.markdown("## ⚙️ Bulk Actions")
if st.sidebar.button("🧠 Bulk generate Text Answers"):
    pending_images = list(st.session_state.image_files)
    if not pending_images:
        st.sidebar.info("No questions in the current selection.")
    else:
        progress = st.sidebar.progress(0, text=f"Starting 0/{len(pending_images)}")
        status = st.sidebar.empty()
        generated_count = 0
        failed_count = 0
        failed_images = []
        for i, image_name in enumerate(pending_images, 1):
            image_path = os.path.join(folder_path, image_name)
            cache_key = canonical_question_cache_key(
                BASE_ROOT,
                image_path,
                question_cache_key(course_level, current_topic_key, current_subtopic_key, image_name),
            )
            last_error = None
            for attempt in range(1, 4):
                status.caption(f"Generating {i}/{len(pending_images)}: {image_name} (attempt {attempt}/3)")
                try:
                    generate_text_answer_for_question(
                        image_path,
                        cache_key,
                        question_type_file,
                        {
                            "user": current_user,
                            "course": course_level,
                            "topic": current_topic_key,
                            "subtopic": current_subtopic_key,
                            "image": image_name,
                            "question_number": i,
                        },
                        force_regen=True,
                    )
                    generated_count += 1
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 3:
                        time.sleep(2 * attempt)
            if last_error is not None:
                failed_count += 1
                failed_images.append(image_name)
            progress.progress(i / len(pending_images), text=f"Processed {i}/{len(pending_images)}")
        if failed_count:
            st.sidebar.warning(f"Generated {generated_count}; failed {failed_count}.")
            with st.sidebar.expander("Failed questions"):
                for image_name in failed_images:
                    st.caption(image_name)
        else:
            st.sidebar.success(f"Generated {generated_count} Text Answers.")

col1, col2 = st.columns([2, 2])

with col2:
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button("⬅️ Previous", disabled=st.session_state.question_index == 0):
            st.session_state.question_index -= 1
            st.session_state.gemini_chat_history = []
    with nav_right:
        if st.button("➡️ Next", disabled=st.session_state.question_index >= len(st.session_state.image_files) - 1):
            st.session_state.question_index += 1
            st.session_state.gemini_chat_history = []

if st.session_state.image_files:
    q_index = st.session_state.question_index
    q_index = min(max(q_index, 0), len(st.session_state.image_files) - 1)
    img_name = st.session_state.image_files[q_index]
    img_path = os.path.join(folder_path, img_name)
    generated_question_key = canonical_question_cache_key(
        BASE_ROOT,
        img_path,
        question_cache_key(course_level, current_topic_key, current_subtopic_key, img_name),
    )
    current_question_type = get_question_type(question_type_file, generated_question_key) if question_type_exists(question_type_file, generated_question_key) else DEFAULT_QUESTION_TYPE
    saved_flash_card = load_saved_answer(generated_question_key, "flash_card")

if show_answer_summary and st.session_state.image_files:
    selected_answer_items = []
    for image_name in st.session_state.image_files:
        answer_path = os.path.join(folder_path, image_name)
        answer_key = canonical_question_cache_key(
            BASE_ROOT,
            answer_path,
            question_cache_key(course_level, current_topic_key, current_subtopic_key, image_name),
        )
        selected_answer_items.append(
            {
                "key": answer_key,
                "image": os.path.basename(image_name),
                "question_type": get_question_type(question_type_file, answer_key)
                if question_type_exists(question_type_file, answer_key)
                else DEFAULT_QUESTION_TYPE,
            }
        )

    answer_summary = build_answer_summary(read_json_list(USER_ANSWER_FILE), selected_answer_items)
    st.markdown("### 📊 Your Answer Summary")
    st.caption(
        f"Answered {answer_summary['answered_count']} of {answer_summary['total_count']} questions in the current selection."
    )
    if answer_summary["rows"]:
        display_rows = []
        for row in answer_summary["rows"]:
            answer_preview = str(row["Answer"]).replace("\\n", " ").strip()
            if len(answer_preview) > 160:
                answer_preview = answer_preview[:157] + "..."
            feedback_preview = str(row.get("Feedback", "")).replace("\\n", " ").strip()
            if len(feedback_preview) > 160:
                feedback_preview = feedback_preview[:157] + "..."
            display_rows.append({**row, "Answer": answer_preview, "Feedback": feedback_preview})
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No student answers saved for this selection yet.")

    with col1:
        st.markdown(f"### 📘 Question {q_index + 1} of {len(st.session_state.image_files)}")
        st.image(img_path, caption=f"🖼️ Question Image {q_index + 1}: {img_name}")
        st.caption(f"🏷️ Type of question: {current_question_type}")

        if (saved_flash_card or "").strip():
            st.markdown("### 🃏 Flash Card")
            render_flash_card(saved_flash_card, f"{generated_question_key}:{q_index}")

        st.markdown("### ✍️ Your Answer")
        answer_widget_id = hashlib.sha1(generated_question_key.encode("utf-8")).hexdigest()
        previous_student_answer = latest_answers_by_key(read_json_list(USER_ANSWER_FILE)).get(generated_question_key)
        previous_answer_text = (previous_student_answer or {}).get("answer", "")

        if current_question_type == "Multiple choice":
            choices = ["A", "B", "C", "D"]
            selected_index = choices.index(previous_answer_text) if previous_answer_text in choices else 0
            student_answer = st.radio("Choose your answer:", choices, index=selected_index, horizontal=True, key=f"student_answer_choice_{answer_widget_id}")
        else:
            student_answer = st.text_area("Write your answer:", value=previous_answer_text, height=140, key=f"student_answer_text_{answer_widget_id}")

        if st.button("💾 Submit Answer", key=f"submit_student_answer_{answer_widget_id}"):
            answer_text = (student_answer or "").strip()
            if current_question_type != "Multiple choice" and not answer_text:
                st.warning("Please write an answer before submitting.")
            else:
                append_answer_log(
                    USER_ANSWER_FILE,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "user": current_user,
                        "course": course_level,
                        "topic": current_topic_key,
                        "subtopic": current_subtopic_key,
                        "question_type": current_question_type,
                        "image": img_name,
                        "key": generated_question_key,
                        "answer": answer_text,
                        "feedback": "",
                        "question_number": q_index + 1,
                    },
                )
                st.success("✅ Answer saved.")

    with col2:
        c1, c2 = st.columns(2)
        has_text_answer = bool(load_saved_answer(generated_question_key, "text"))
        has_graph_answer = bool(load_saved_answer(generated_question_key, "graph"))
        clicked_text = c1.button("🧠 Show Text Answer" if has_text_answer else "🧠 Answer with Text", key=f"explain_{q_index}")
        clicked_text_regen = c1.button("🔄 Regenerate Text", key=f"regen_text_{q_index}")
        clicked_graph = c2.button("📈 Show Graph Answer" if has_graph_answer else "📈 Answer with Graph", key=f"graph_{q_index}")
        clicked_graph_regen = c2.button("🔄 Regenerate Graph", key=f"regen_graph_{q_index}")

        if clicked_text or clicked_text_regen:
            if has_text_answer and not clicked_text_regen:
                reply = load_saved_answer(generated_question_key, "text")
                question_type = current_question_type
            else:
                with st.spinner("LLM is thinking ... ... 👩‍✨"):
                    reply, question_type = generate_text_answer_for_question(
                        img_path,
                        generated_question_key,
                        question_type_file,
                        {
                            "user": current_user,
                            "course": course_level,
                            "topic": current_topic_key,
                            "subtopic": current_subtopic_key,
                            "image": img_name,
                            "question_number": q_index + 1,
                        },
                        force_regen=clicked_text_regen,
                    )
            st.session_state.visible_text_answers[generated_question_key] = {"reply": reply, "question_type": question_type}

        if clicked_graph or clicked_graph_regen:
            with st.spinner("LLM is thinking ... ... 👩‍✨"):
                raw = None if clicked_graph_regen else load_saved_answer(generated_question_key, "graph")
                if not (raw or "").strip():
                    response = call_model(GRAPH_ANSWER_PROMPT, load_image_for_model(img_path))
                    raw = response.text
                    save_answer(generated_question_key, "graph", raw)
                st.session_state.visible_graph_answers[generated_question_key] = raw

        video_col, flash_col = st.columns(2)
        if video_col.button("🎮 Video Help", key=f"video_{q_index}"):
            video_prompt = """
You are an expert NSW HSC Mathematics teacher. Based on the image below, recommend one or two YouTube tutorial resources that help explain the concepts or topic shown.

Please format like:
- [Video Title](https://youtube.com/...)
  Short reason this matches the question.
"""
            response = call_model(video_prompt, load_image_for_model(img_path))
            st.markdown("### 🎥 Recommended Video")
            st.markdown(response.text.strip())

        if flash_col.button("🃏 Flash Card", key=f"flash_card_{q_index}"):
            saved_text_answer = load_saved_answer(generated_question_key, "text")
            if (saved_flash_card or "").strip():
                st.info("Saved flash card is shown under the question.")
            elif not (saved_text_answer or "").strip():
                st.warning("No saved Text Answer is available yet. Please show or generate the Text Answer first.")
            else:
                with st.spinner("Creating flash card..."):
                    response = call_text_model(build_flash_card_prompt(strip_question_type_section(saved_text_answer)))
                saved_flash_card = response.text.strip()
                save_answer(generated_question_key, "flash_card", saved_flash_card)
                st.rerun()

        if (reply := st.session_state.visible_text_answers.get(generated_question_key)):
            render_text_answer_card(reply["reply"], reply["question_type"], f"{generated_question_key}:{q_index}:text")
        if (raw_graph := st.session_state.visible_graph_answers.get(generated_question_key)):
            display_graph_answer(raw_graph, os.path.splitext(os.path.basename(img_name))[0])

    if selected_note_pdf:
        note_path = os.path.join(NOTES_ROOT, selected_note_pdf)
        if os.path.exists(note_path):
            with st.expander(f"📚 Note: {selected_note_pdf}", expanded=True):
                show_pdf_page_viewer(note_path, f"{course_level}_{selected_note_pdf}")
        else:
            st.sidebar.caption("The selected note PDF is not available yet.")
else:
    st.warning("⚠️ No PNG images found in the selected sub-topic.")

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:16px;'>👧 Keep going! Every question makes you stronger 💪 and smarter 🧠.</p>", unsafe_allow_html=True)

st.stop()

############################################
# ---------- Main layout ----------
############################################
col1, col2 = st.columns([2, 2])

with col2:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬅️ Previous", disabled=st.session_state.question_index == 0):
            st.session_state.questions.pop(st.session_state.question_index, None)
            st.session_state.question_index -= 1
            st.session_state.gemini_chat_history = []
    with b2:
        if st.button("➡️ Next", disabled=st.session_state.question_index >= len(st.session_state.image_files) - 1):
            st.session_state.questions.pop(st.session_state.question_index, None)
            st.session_state.question_index += 1
            st.session_state.gemini_chat_history = []
    if "question_number" not in st.session_state or st.session_state.question_number != st.session_state.question_index + 1:
        st.session_state.question_number = st.session_state.question_index + 1

# progress
total_imgs = len(st.session_state.image_files)
if total_imgs == 0 or st.session_state.question_index >= total_imgs:
    st.warning("⚠️ No more questions available.")
else:
    st.progress((st.session_state.question_index + 1) / total_imgs,
                text=f"🌟 You're working on question {min(st.session_state.question_index + 1, total_imgs)}!")

if st.session_state.image_files:
    q_index = st.session_state.question_index
    if 0 <= q_index < len(st.session_state.image_files):
        img_name = st.session_state.image_files[q_index]
        img_path = os.path.join(folder_path, img_name)

        with col1:
            total_questions = len(st.session_state.image_files)
            current_question = st.session_state.question_index + 1
            st.markdown(f"### 📘 Question {current_question} of {total_questions}")
            st.image(img_path, caption=f"🖼️ Question Image {q_index+1}: {img_name}")
        # Explain / Video / Generate (in col2)
        with col2:
            c1, c2 = st.columns(2)
            clicked_explain = c1.button("🧠 Answer with Text", key=f"explain_{q_index}")
            clicked_regen   = c2.button("🔄 Answer with Graph", key=f"regen_{q_index}")

            if clicked_explain:
                with st.spinner("LLM is thinking ... ... 👩‍✨"):
                    import base64, sys, importlib.util

                    base_no_ext = os.path.splitext(os.path.basename(img_name))[0]
                    json_filename = f"{base_no_ext}.explain.json"
                    explain_path = os.path.join(folder_path, json_filename)
                    
                    prompt_answer = """
    You are a top HSC teacher. Read the image and question below, then ANSWER the question and EXPLAIN how to solve it.
    ⚠️ IMPORTANT: Please format all LaTeX math expressions using:
    - `$...$` for inline math
    - `$$...$$` for block math

    Do not use `\\(...\\)` or `\\[...\\]`.
"""
                    # Try to load existing JSON to avoid re-running the LLM
                    data = None
                    # If no cached JSON, call LLM and save
                    if data is None:
                        with open(img_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        image = Image.open(BytesIO(img_bytes))

                        response = call_model(prompt_answer, image)
                        st.markdown("#### ✅ Answer")
                        reply = response.text
                        st.markdown(reply)
                    else:
                        st.info("No 'Answer' field returned.")
            if clicked_regen:
                with st.spinner("LLM is thinking ... ... 👩‍✨"):
                    import base64, sys, importlib.util

                    force_regen = clicked_regen

                    base_no_ext = os.path.splitext(os.path.basename(img_name))[0]
                    txt_filename  = f"{base_no_ext}.explain.json"
                    explain_path = os.path.join(folder_path, txt_filename )
                    
                    prompt_answer = """
                        You are a top HSC teacher.

                        ⬇️ RETURN PLAIN TEXT ONLY (NO JSON).  
                        Format your answer using these exact section headers:

                        ###ANSWER
                        (write the final answer only, concise)

                        ###PLOT_CODE (optional)
                        # Python code that defines generate_plot() and returns a Plotly or Matplotlib figure as `fig`
                        ###EXPLANATION
                        (step-by-step reasoning, key HSC concepts)

                        ###OTHERS
                        (any extra insights)

                        ⚠️ Do NOT return JSON. Do NOT escape characters. Just plain text.
                        """
                    # Load from cache if available
                    data = {"ANSWER": "", "PLOT_CODE": "", "EXPLANATION": "", "OTHERS": ""}
                    loaded_from_cache = False

                    if (not force_regen) and os.path.exists(explain_path):
                        with open(explain_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        loaded_from_cache = True
                        st.success(f"Loaded saved explanation: {txt_filename}")
                    else:
                        # call LLM
                        with open(img_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        image = Image.open(BytesIO(img_bytes))

                        response = call_model(prompt_answer, image)
                        raw = response.text

                        with open(explain_path, "w", encoding="utf-8") as f:
                            f.write(raw)
                        st.info(f"💾 Explanation saved to: {explain_path}")
                    
                    # ---- Parse sections ----
                    def extract_section(text, key):
                        """
                        Extract a section using ###KEY and removes ``` fences if present.
                        """
                        marker = f"###{key}"
                        if marker not in text:
                            return ""

                        part = text.split(marker, 1)[1]

                        # Stop section at next heading
                        for next_header in ["###ANSWER", "###PLOT_CODE", "###EXPLANATION", "###OTHERS"]:
                            if next_header != marker and next_header in part:
                                part = part.split(next_header, 1)[0]

                        cleaned = part.strip()

                        # ✅ Strip code fences if LLM wrapped it
                        if cleaned.startswith("```"):
                            cleaned = cleaned.lstrip("`")  # remove opening backticks
                            cleaned = cleaned.lstrip("python")  # if format is ```python
                            cleaned = cleaned.strip()  # remove whitespace
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3].rstrip()

                        return cleaned

                    data["ANSWER"]       = extract_section(raw, "ANSWER")
                    data["PLOT_CODE"]    = extract_section(raw, "PLOT_CODE")
                    data["EXPLANATION"]  = extract_section(raw, "EXPLANATION")
                    data["OTHERS"]       = extract_section(raw, "OTHERS")
                    
                    # ---- Display ----
                    if data["ANSWER"]:
                        st.markdown("#### ✅ Answer")
                        st.markdown(data["ANSWER"])

                    # plot / code
                    if data["PLOT_CODE"]:
                        try:
                            explain_py = os.path.join(user_tmp_dir, f"explain_plot_{base_no_ext}.py")
                            with open(explain_py, "w", encoding="utf-8") as f:
                                f.write(data["PLOT_CODE"])

                            mod = load_plot_module(explain_py)
                            fig = getattr(mod, "generate_plot", lambda: None)()
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Unable to execute plot code: {e}")

                    if data["EXPLANATION"]:
                        st.markdown("#### 📘 Detailed Explanation")
                        st.markdown(data["EXPLANATION"])

                    if data["OTHERS"]:
                        st.markdown("#### 🧩 Other Information")
                        st.markdown(data["OTHERS"])
        with col2:
            if st.button("🎮 Video Help", key=f"video_{q_index}"):
                video_prompt = """
You are an expert HSC teacher. Based on the image below, recommend one or two specific YouTube video tutorials (with full URLs) that help explain the concepts or topic shown. Include a short description of each video.
Please format like:
- [Video Title](https://youtube.com/...)
"""
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                image = Image.open(BytesIO(img_bytes))
                response = call_model(video_prompt, image)
                st.markdown("### 🎥 Recommended Video")
                st.markdown(response.text.strip())

        with col2:
            if st.button("✨ Generate Question"):
                with st.spinner("Generating your magical question... 👩‍✨"):
                    img_name = st.session_state.image_files[q_index]
                    img_path = os.path.join(folder_path, img_name)

                    with open(img_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    image = Image.open(BytesIO(img_bytes))
                    
                    try:
                        response_text = call_model(prompt_generating, image)
                        data = extract_json_from_response(response_text)
                        st.session_state.questions[q_index] = data
                        tk_dir = os.path.join(user_tmp_dir, "TK_questions")
                        os.makedirs(tk_dir, exist_ok=True)
                        out_json = os.path.join(tk_dir, f"{selected_subtopic}_Q{q_index+1:02d}.json")
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        if "Image_DataTable" in data:
                            image_py = os.path.join(user_tmp_dir, "image.py")
                            with open(image_py, "w", encoding="utf-8") as f:
                                f.write(data["Image_DataTable"])
                            st.session_state["_image_py_path"] = image_py
                        st.success(f"Saved generated question to {out_json}")
                    except Exception as e:
                        st.error(f"❌ Oops! Something went wrong: {str(e)}")
        with col2:
            if q_index in st.session_state.questions:
                question = st.session_state.questions[q_index]
                st.markdown(f"### ❓ Question {q_index + 1}")
                st.markdown(f"💬 {question.get('Question', '')}")
                q_type = question.get("Question_Type", "MultipleChoice")

                # load per-user image.py if exists
                image_py = st.session_state.get("_image_py_path", os.path.join(user_tmp_dir, "image.py"))
                if os.path.exists(image_py):
                    try:
                        fig = load_plot_module(image_py).generate_plot()
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Unable to load chart: {e}")

                # Answer inputs
                if q_type == "MultipleChoice":
                    options = question.get("Multiple Choices", [])
                    if options:
                        selected = st.radio("🌟 Choose your answer:", options, key=f"radio_{q_index}")
                        st.session_state.user_answers[q_index] = selected
                elif q_type in ["OpenEnded", "Proof", "ShortAnswer", "TableCompletion"]:
                    st.session_state.user_answers[q_index] = st.text_area("✍️ Write your answer:", key=f"textarea_{q_index}")
                else:
                    st.info("ℹ️ Unrecognised question type — capturing free-form answer.")
                    st.session_state.user_answers[q_index] = st.text_area("✍️ Your answer:", key=f"textarea_{q_index}")

                if st.button("✅ Submit Answer", key=f"submit_{q_index}"):
                    user_ans = st.session_state.user_answers.get(q_index)
                    correct = question.get("Answer", "")
                    if q_type == "MultipleChoice":
                        if user_ans == correct:
                            st.balloons()
                            st.success("🎉 Correct! You nailed it.")
                        else:
                            st.error(f"😢 Not quite. The correct answer is: {correct}")
                    else:
                        st.info(f"✅ Your answer is saved. Here's the marking guide or solution:\n\n**{correct}**")

                current_question_key = f"{current_course_key}/{current_topic_key}/{current_subtopic_key}/{img_name}"
                saved_flash_card = load_saved_answer(current_question_key, "flash_card")
                flash_card_requested = st.button("🃏 Flash Card", key=f"flash_card_{q_index}")

                if flash_card_requested:
                    if (saved_flash_card or "").strip():
                        st.info("Saved flash card loaded below.")
                    elif not (question.get("Answer", "") or "").strip():
                        st.warning("No answer is available yet. Generate the question first so I can build a flash card.")
                    else:
                        with st.spinner("Creating flash card..."):
                            response = call_text_model(build_flash_card_prompt(question.get("Answer", "")))
                        saved_flash_card = response.text.strip()
                        save_answer(current_question_key, "flash_card", saved_flash_card)
                        st.success("Flash card saved.")
                        st.rerun()

                if (saved_flash_card or "").strip():
                    with st.expander("🃏 Flash Card Review", expanded=True):
                        st.caption("Click the card to flip it. The card is saved in your per-user study cache.")
                        render_flash_card(saved_flash_card, f"{current_question_key}:{q_index}")

        # 📝 Feedback & Notes Section (append-only log + auto-load previous notes)
        with col1:
            st.markdown("### 📝 Your Feedback")
            feedback_key = f"{course_level}/{selected_topic}/{selected_subtopic}/{img_name}"
            # --- Load ALL previous notes immediately when question loads ---
            history = get_all_feedback_for_key(FEEDBACK_FILE, feedback_key)
            if history:
                with st.expander("📚 Previous notes & statuses (this question)", expanded=True):
                    for i, h in enumerate(history, 1):
                        st.markdown(
                            f"**{i}. {h.get('ts','')}** — **{h.get('status','')}**<br/>"
                            f"{(h.get('note') or '').strip()}",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No previous notes yet for this question.")

            # Use the most recent entry to prefill
            last = history[-1] if history else None

            feedback_options = {
                "good": "✅ I am good on this question",
                "review": "🔄 I need to review this question",
                "challenge": "❗ It is challenging and I need more understanding",
            }

            default_index = 1
            if last and last.get("status") in feedback_options:
                default_index = list(feedback_options.keys()).index(last["status"])

            selected_feedback = st.radio(
                "🗣️ How do you feel about this question?",
                list(feedback_options.keys()),
                format_func=lambda x: feedback_options[x],
                index=default_index,
                key=f"feedback_{q_index}",
            )

            user_note = st.text_area(
                "📝 Your personal note:",
                value=last.get("note", "") if last else "",
                key=f"note_{q_index}",
            )

            if st.button("💾 Save Feedback (Append)", key=f"save_{q_index}"):
                entry = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "user": current_user,
                    "course": course_level,
                    "topic": selected_topic,
                    "subtopic": selected_subtopic,
                    "image": img_name,
                    "key": feedback_key,
                    "status": selected_feedback,
                    "question_number": st.session_state.question_number,  # or another source
                    "note": user_note,
                }
                append_json_log(FEEDBACK_FILE, entry)
                st.success("✅ Logged! (Appended to feedback file)")

        with col2:
            # 🔮 Gemini Chatbox on Main Page (latest on top)
            st.markdown("### 🤖 Chat with Gemini about this Question")

            if "gemini_chat_history" not in st.session_state:
                st.session_state.gemini_chat_history = []

            with st.expander("💬 Open Chat with Gemini", expanded=True):
                user_input = st.chat_input("Ask a question about the image, explanation, or topic...")
                if user_input:
                    # Show user message
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    st.session_state.gemini_chat_history.append(("user", user_input))

                    # Get image for context
                    if st.session_state.image_files and 0 <= q_index < len(st.session_state.image_files):
                        img_name = st.session_state.image_files[q_index]
                        img_path = os.path.join(folder_path, img_name)
                        with open(img_path, "rb") as f:
                            image = Image.open(BytesIO(f.read()))

                        model = genai.GenerativeModel(selected_model)
                        with st.spinner("Gemini is thinking..."):
                            try:
                                response = model.generate_content([user_input, image])
                                reply = response.text
                            except Exception as e:
                                reply = f"❌ Error: {e}"
                    else:
                        reply = "⚠️ No image context available."

                    # Show assistant response
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                    st.session_state.gemini_chat_history.append(("assistant", reply))

                    st.rerun()  # Refresh UI to show new message at top
                
                # Show latest messages at the top
                for role, message in reversed(st.session_state.gemini_chat_history):
                    with st.chat_message(role):
                        st.markdown(message)
else:
    st.warning("⚠️ No PNG images found in the selected sub-topic.")

import pandas as pd

if mode == "Past Paper":
    if st.sidebar.button("📊 Show Past Paper Summary"):
        st.markdown("## 📊 Past Paper Feedback Summary")

        # Load all feedback entries for this user and course
        all_feedback = read_json_list(FEEDBACK_FILE)

        # Filter for only past paper entries (topic == "PastPaper")
        current_paper = st.session_state.get("selected_paper", "")
        past_paper_feedback = [
            fb for fb in all_feedback
            if fb.get("topic") == "PastPaper" and fb.get("subtopic") == current_paper.replace(" ", "_")
        ]

        if not past_paper_feedback:
            st.info("ℹ️ No past paper feedback available yet.")
        else:
            # Build a nested dictionary: { paper: { Q#: feedback_status } }
            summary = {}
            for fb in past_paper_feedback:
                paper = fb.get("subtopic", "Unknown Paper")
                image = fb.get("image", "")
                q_num = fb.get("question_number")
                if not q_num:
                    continue  # skip if missing

                feedback_status = fb.get("status", "-")

                if paper not in summary:
                    summary[paper] = {}
                summary[paper][q_num] = feedback_status

            # Determine the max number of questions for column layout
            max_q = max((max(qs.keys()) for qs in summary.values()), default=0)
            df = pd.DataFrame([
                {"Past Paper": paper, **{f"Q{q}": status for q, status in questions.items()}}
                for paper, questions in summary.items()
            ])

            # Reorder columns: put 'Past Paper' first, then Q1, Q2, ...
            desired_columns = ["Past Paper"] + [f"Q{i}" for i in range(1, max_q + 1)]
            df = df.reindex(columns=desired_columns, fill_value="-")

            df.fillna("-", inplace=True)

            # Style feedback cells
            def style_feedback(val):
                color_map = {
                    "good": "#a7f3d0",
                    "review": "#fde68a",
                    "challenge": "#fca5a5",
                }
                color = color_map.get(val, "")
                if color:
                    return f"background-color: {color}; color: transparent;"  # Hide text
                return ""

            # Assumes: df contains columns like "Past Paper", "Q1", ..., "Q20"
            chunk_size = 10

            # Extract only question columns (Q1, Q2, ...)
            question_cols = [col for col in df.columns if col.startswith("Q")]
            chunks = [question_cols[i:i + chunk_size] for i in range(0, len(question_cols), chunk_size)]

            for chunk in chunks:
                display_cols = ["Past Paper"] + chunk
                sub_df = df[display_cols]

                styled_chunk = sub_df.style.map(style_feedback, subset=chunk)

                # Optional: Add section header
                q_start = chunk[0]
                q_end = chunk[-1]
                
                st.table(styled_chunk)

            # Add legend
            st.markdown("""
                <div style='margin-top:10px;'>
                <strong>Legend:</strong><br>
                <span style='background-color:#a7f3d0;padding:4px;'>✅ good</span>
                <span style='background-color:#fde68a;padding:4px;'>🔄 review</span>
                <span style='background-color:#fca5a5;padding:4px;'>❗ challenge</span>
                <span style='padding:4px;'>➖ no feedback</span>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:16px;'>👧 Keep going! Every question makes you stronger 💪 and smarter 🧠.</p>", unsafe_allow_html=True)
