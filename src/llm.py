import streamlit as st
import os, re, json
from PIL import Image
from io import BytesIO
from src.config import GEMINI_API_KEY
import google.generativeai as genai


prompt_template = """
You are a top HSC teacher. Based on the image below, generate a high-quality NSW HSC-style question:
Return format:
{
"Question": "...",
"Question_Type": "MultipleChoice",
"Image_DataTable": "Python code block to generate visuals",
"Multiple Choices": ["A", "B", "C", "D"],
"Answer": "B",
"Marks": 2
}
"""


def setup_gemini():
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY")
        st.stop()
    genai.configure(api_key=GEMINI_API_KEY)


def extract_json(text):
    m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if not m:
        m = re.search(r"(\{[\s\S]+\})", text)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(1))


def call_gemini(prompt, image: Image.Image):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([prompt, image])
    return response


def handle_question_logic(user):
    st.info("(🔧 This section will be connected to your image/question logic here.)")
    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)
        if st.button("Generate HSC Question"):
            resp = call_gemini(prompt_template, image)
            try:
                data = extract_json(resp.text)
                st.json(data)
            except Exception as e:
                st.error(f"Failed to extract JSON: {e}")