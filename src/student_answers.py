import json
import html
import os
import re


def read_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_answer_log(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = read_json_list(path)
    entries.append(dict(entry))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def latest_answers_by_key(entries: list) -> dict:
    latest = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not key:
            continue
        current = latest.get(key)
        if current is None or entry.get("timestamp", "") >= current.get("timestamp", ""):
            latest[key] = entry
    return latest


def build_answer_summary(entries: list, selection: list) -> dict:
    latest = latest_answers_by_key(entries)
    rows = []
    for index, item in enumerate(selection, 1):
        key = item.get("key")
        entry = latest.get(key)
        if entry is None:
            continue
        rows.append(
            {
                "Question": index,
                "Image": item.get("image", ""),
                "Type": entry.get("question_type") or item.get("question_type", ""),
                "Last answered": entry.get("timestamp", ""),
                "Answer": entry.get("answer", ""),
                "Feedback": entry.get("feedback", ""),
            }
        )
    return {
        "answered_count": len(rows),
        "total_count": len(selection),
        "rows": rows,
    }


def build_answer_feedback_prompt(question_type: str, student_answer: str, teacher_answer: str) -> str:
    return f"""
You are a supportive NSW HSC Physics tutor. Compare the student's answer with the saved teacher answer.

Question type: {question_type}

Student answer:
{student_answer}

Saved teacher answer:
{teacher_answer}

Give concise, encouraging feedback in this format:
- Result: correct, partly correct, or needs work
- What was good
- What to fix or add
- A short improved answer the student could write

Keep it student-friendly and do not be harsh.
""".strip()


def build_flash_card_prompt(saved_answer: str) -> str:
    return f"""
You are a concise NSW HSC Physics study coach. Create one flash card from the saved answer below.

Saved answer:
{saved_answer}

Format exactly:
### Front
A short recall question about the key physics law, formula, constant, figure, graph feature, or principle.

### Back
- Key idea:
- Formula / law / constant / figure:
- When to use it:
- Common trap:

Keep it compact and exam-focused. If no numeric constant or fixed figure applies, write "No fixed constant or figure".
""".strip()


def parse_flash_card(text: str) -> dict:
    front_match = re.search(r"###\s*Front\s*([\s\S]*?)(?=###\s*Back|$)", text or "", re.IGNORECASE)
    back_match = re.search(r"###\s*Back\s*([\s\S]*)", text or "", re.IGNORECASE)
    return {
        "front": front_match.group(1).strip() if front_match else "Review this key HSC Physics idea.",
        "back": back_match.group(1).strip() if back_match else (text or "").strip(),
    }


def _flash_card_inline_markdown_to_html(text: str) -> str:
    rendered = html.escape(text or "")
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
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


def flash_card_markdown_to_html(text: str) -> str:
    return study_markdown_to_html(text)


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


def canonical_question_cache_key(base_root: str, image_path: str, fallback_key: str) -> str:
    try:
        base_abs = os.path.abspath(base_root)
        image_abs = os.path.abspath(image_path)
        common = os.path.commonpath([base_abs, image_abs])
        if common != base_abs:
            return fallback_key
        rel_path = os.path.relpath(image_abs, base_abs)
    except Exception:
        return fallback_key

    rel_parts = rel_path.split(os.sep)
    if len(rel_parts) < 4 or rel_path.startswith(".."):
        return fallback_key
    return rel_path.replace(os.sep, "/")


def question_type_course_for_cache_key(cache_key: str, fallback_course: str) -> str:
    first_part = (cache_key or "").split("/", 1)[0]
    if re.match(r"^M\d+\.", first_part or ""):
        return first_part
    return fallback_course
