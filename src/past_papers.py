import os
import re


def paper_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"_(?:Picture|Group)\s+\d+$", "", stem)


def matches_selected_paper(filename: str, selected_paper: str) -> bool:
    return paper_name_from_filename(filename) == (selected_paper or "").strip()


def extract_picture_number(filename: str) -> int:
    match = re.search(r"(?:Picture|Group)\s(\d+)\.png$", filename)
    return int(match.group(1)) if match else 0
