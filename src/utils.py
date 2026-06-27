# ========================= src/utils.py =========================
import os
import json
import re
import base64
from typing import Optional
from PIL import Image
from io import BytesIO

def read_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]  # support older formats
        return []
    except Exception:
        return []

def append_json_log(path: str, entry: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = read_json_list(path)
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_json_from_text(text: str) -> Optional[dict]:
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if not match:
        match = re.search(r"(\{[\s\S]+?\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def image_to_b64_png(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def extract_plot_code(data: dict) -> str:
    b64 = (data.get("Image_DataTable_b64") or "").strip()
    if b64:
        try:
            return base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            pass
    return (data.get("Image_DataTable") or "").strip()
