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

############################################
# 💼 Multi-user Auth + Per-user Storage (Streamlit Cloud ready)
############################################
APP_TITLE = "HSC AI Tutoring Centre"

# Use secrets or env to configure base paths for deployment
PERSIST_DIR = os.getenv("PERSIST_DIR") or st.secrets.get("PERSIST_DIR", "persist")
BASE_ROOT = os.getenv("BASE_ROOT") or st.secrets.get("BASE_ROOT", "data/HSCMath")

ACCOUNTS_DB = os.path.join(PERSIST_DIR, "server", "users.json")
USERS_ROOT = os.path.join(PERSIST_DIR, "users")

# ensure required dirs
os.makedirs(os.path.dirname(ACCOUNTS_DB), exist_ok=True)
os.makedirs(USERS_ROOT, exist_ok=True)

# password hashing helpers
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


############################################
# 🌟 App Settings
############################################
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --------- Auth UI ---------
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

st.markdown("<h1 style='text-align: center;'> Welcome to HSC Math Centre ✨</h1>", unsafe_allow_html=True)

with st.expander("🔐 Sign in (Auto sign-up if no account)", expanded=st.session_state.auth_user is None):
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
            db = load_users()
            user = db["users"].get(username)
            if user is None:
                # auto sign-up
                salt = _new_salt()
                pw_hash = _hash_pw(password, salt)
                db["users"][username] = {
                    "salt": salt,
                    "hash": pw_hash,
                    "created": datetime.utcnow().isoformat() + "Z",
                }
                save_users(db)
                ensure_user_space(username)
                st.success("Account created and signed in ✨")
                st.session_state.auth_user = username
            else:
                calc = _hash_pw(password, user["salt"])
                if calc == user["hash"]:
                    ensure_user_space(username)
                    st.success("Signed in ✅")
                    st.session_state.auth_user = username
                else:
                    st.error("Incorrect password. Please try again.")

# sign-out
if st.session_state.auth_user:
    st.sidebar.success(f"Signed in as **{st.session_state.auth_user}**")
    if st.sidebar.button("Sign out"):
        st.session_state.auth_user = None
        st.rerun()

if st.session_state.auth_user is None:
    st.stop()

current_user = st.session_state.auth_user
user_root, user_fb_dir, user_tmp_dir = ensure_user_space(current_user)

st.markdown(
    f"<p style='text-align: center;'>Be a star today, {current_user}! ⭐ Your data is saved in <code>{user_root}</code></p>",
    unsafe_allow_html=True,
)

############################################
# 🧠 Gemini API Setup
############################################
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set in environment or secrets")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

############################################
# 🎯 Prompt Template
############################################
prompt_template = """
You are a top HSC teacher. Based on the concepts shown in the input image, generate a high-quality Y12 NSW HSC-style question (not limited to multiple choice). Follow this structure:
{
  "Question": "The question in text format, including mathematical notation if applicable.",
  "Question_Type": "one of: 'OpenEnded', 'Proof', 'MultipleChoice', 'ShortAnswer', 'TableCompletion', etc.",
  "Image_DataTable": "Python code to generate any diagram or data table, using Plotly or Matplotlib. Must define a function `generate_plot()` that returns a figure.",
  "Multiple Choices": ["A", "B", "C", "D"],
  "Answer": "Correct answer or marking guide.",
  "Marks": "Total marks allocated, if applicable."
}
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


# Sort by numeric suffix before .png

def extract_picture_number(filename):
    match = re.search(r'(?:Picture|Group)\s(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0


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
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


############################################
# ---------- Sidebar: Course & Mode Selection ----------
############################################
st.sidebar.markdown("## 🔍 Select HSC Math Exam Level")
course_level = st.sidebar.selectbox("🎓 Course:", ["Math_3U", "Math_2U", "Physics" , "Economics"], key="course_level")

st.sidebar.markdown("## 🧭 Select Practice Mode")
mode = st.sidebar.radio("Choose practice mode:", ["Topic by Topic", "Past Paper"], key="practice_mode")

base_root = os.path.join(BASE_ROOT, course_level)
if not os.path.isdir(base_root):
    st.warning(f"⚠️ The path '{base_root}' was not found. Please check your course folder structure.")
    st.stop()

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

# Determine full image path depending on mode
if mode == "Past Paper":
    folder_path = st.session_state.folder  # base QuestionLibrary
    selected_topic = "PastPaper"
    selected_subtopic = st.session_state.selected_paper.replace(" ", "_")
else:
    selected_topic = selected_topic if 'selected_topic' in locals() else st.session_state.last_selected_topic
    selected_subtopic = selected_subtopic if 'selected_subtopic' in locals() else st.session_state.last_selected_subtopic
    folder_path = os.path.join(base_root, selected_topic, selected_subtopic)

############################################
# ---------- Selection Changed ----------
############################################
selection_changed = (
    st.session_state.last_selected_course != course_level or
    st.session_state.last_selected_topic != st.session_state.get("last_selected_topic") or
    st.session_state.last_selected_subtopic != st.session_state.get("last_selected_subtopic") or
    st.session_state.last_mode != mode or
    st.session_state.last_paper != st.session_state.get("selected_paper")
)

if selection_changed:
    st.session_state.question_index = 0
    st.session_state.last_selected_course = course_level
    st.session_state.last_selected_topic = st.session_state.get("last_selected_topic")
    st.session_state.last_selected_subtopic = st.session_state.get("last_selected_subtopic")
    st.session_state.last_mode = mode
    st.session_state.last_paper = st.session_state.get("selected_paper")
    st.session_state.thinking_map_dot = ""

############################################
# ---------- Per-user, per-course feedback LOG (append-only) ----------
############################################
FEEDBACK_FILE = os.path.join(user_fb_dir, f"question_feedback_{course_level}.json")
st.sidebar.caption(f"🗂️ Feedback log (per-user, append-only): **{FEEDBACK_FILE}**")

############################################
# ---------- Main layout ----------
############################################
col1, col2, col3 = st.columns([3, 2, 1])

with col2:
    if st.button("⬅️ Previous", disabled=st.session_state.question_index == 0):
        st.session_state.questions.pop(st.session_state.question_index, None)
        st.session_state.question_index -= 1

    if st.button("➡️ Next", disabled=st.session_state.question_index >= len(st.session_state.image_files) - 1):
        st.session_state.questions.pop(st.session_state.question_index, None)
        st.session_state.question_index += 1

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
            st.image(img_path, caption=f"🖼️ Question Image {q_index+1}: {img_name}")

        # Explain / Video / Generate (in col2)
        with col2:
            if st.button("🧠 Explain", key=f"explain_{q_index}"):
                prompt = "You are a HSC expert teacher. Read the image and question below, then explain how to solve it."
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                image = Image.open(BytesIO(img_bytes))
                response = model.generate_content([prompt, image])
                st.markdown("### 💡 Explanation")
                st.markdown(response.text.strip())

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
                response = model.generate_content([video_prompt, image])
                st.markdown("### 🎥 Recommended Video")
                st.markdown(response.text.strip())

        with col2:
            if st.button("✨ Generate Question"):
                with st.spinner("Generating your magical question... 👩‍✨"):
                    with open(img_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    image = Image.open(BytesIO(img_bytes))
                    try:
                        response_text = model.generate_content([prompt_template, image])
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

        # 📝 Feedback & Notes Section (append-only log + auto-load previous notes)
        with col2:
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
                    "note": user_note,
                }
                append_json_log(FEEDBACK_FILE, entry)
                st.success("✅ Logged! (Appended to feedback file)")

else:
    st.warning("⚠️ No PNG images found in the selected sub-topic.")

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:16px;'>👧 Keep going! Every question makes you stronger 💪 and smarter 🧠.</p>", unsafe_allow_html=True)
