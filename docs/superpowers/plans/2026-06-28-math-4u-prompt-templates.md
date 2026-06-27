# Math 4U Prompt Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the application's non-video LLM instructions with centralised, student-friendly NSW HSC Mathematics Extension 2 prompts and a working five-label question classification contract.

**Architecture:** Create `src/math4u_prompts.py` as the single source of truth for prompt text, question labels, normalization, and prompt builders. Existing consumers in `app.py`, `src/llm.py`, and `src/student_answers.py` will import from this module while preserving their existing response parsers and public function names.

**Tech Stack:** Python 3, Streamlit, built-in `unittest`, Gemini prompt strings

---

## File Structure

- Create `src/math4u_prompts.py`: owns Math 4U labels, normalization, prompt constants, and prompt builders.
- Create `tests/test_math4u_prompts.py`: verifies classification, output contracts, input preservation, subject language, and video-prompt exclusion.
- Modify `app.py`: consumes central prompts, removes duplicated fallback prompt text, and uses the new multiple-choice label.
- Modify `src/llm.py`: consumes the central question-generation prompt.
- Modify `src/student_answers.py`: re-exports the central feedback and flash-card builders and uses a Math 4U fallback label.

### Task 1: Add Prompt Contract Tests

**Files:**
- Create: `tests/test_math4u_prompts.py`

- [ ] **Step 1: Write the failing prompt-contract tests**

```python
import unittest

from src.math4u_prompts import (
    CLASSIFY_QUESTION_TYPE_PROMPT,
    DEFAULT_QUESTION_TYPE,
    GRAPH_ANSWER_PROMPT,
    QUESTION_GENERATION_PROMPT,
    QUESTION_TYPES,
    TEXT_ANSWER_PROMPT,
    build_answer_feedback_prompt,
    build_flash_card_prompt,
    normalize_question_type,
)


class Math4UPromptTests(unittest.TestCase):
    def test_question_types_are_the_approved_labels(self):
        self.assertEqual(
            QUESTION_TYPES,
            [
                "Multiple Choice Question",
                "Calculation Question",
                "Proof Question",
                "Geometry Question",
                "Others",
            ],
        )
        self.assertEqual(DEFAULT_QUESTION_TYPE, "Others")

    def test_normalization_accepts_legacy_aliases(self):
        cases = {
            "Multiple choice": "Multiple Choice Question",
            "MultipleChoice": "Multiple Choice Question",
            "calculation": "Calculation Question",
            "numerical": "Calculation Question",
            "prove": "Proof Question",
            "show that": "Proof Question",
            "geometric": "Geometry Question",
            "Other questions": "Others",
            "unexpected output": "Others",
        }
        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(normalize_question_type(raw_value), expected)

    def test_classification_prompt_uses_only_approved_labels(self):
        for label in QUESTION_TYPES:
            self.assertIn(label, CLASSIFY_QUESTION_TYPE_PROMPT)
        self.assertNotIn("Experimental questions", CLASSIFY_QUESTION_TYPE_PROMPT)
        self.assertNotIn("Data analysis questions", CLASSIFY_QUESTION_TYPE_PROMPT)

    def test_worked_answer_prompt_preserves_parser_contract(self):
        self.assertIn("NSW HSC Mathematics Extension 2", TEXT_ANSWER_PROMPT)
        self.assertIn("###QUESTION_TYPE", TEXT_ANSWER_PROMPT)
        self.assertIn("$...$", TEXT_ANSWER_PROMPT)
        self.assertIn("$$...$$", TEXT_ANSWER_PROMPT)

    def test_graph_prompt_preserves_section_contract(self):
        for header in ("###ANSWER", "###PLOT_CODE", "###EXPLANATION", "###OTHERS"):
            self.assertIn(header, GRAPH_ANSWER_PROMPT)
        self.assertIn("def generate_plot()", GRAPH_ANSWER_PROMPT)

    def test_question_generation_prompt_requires_valid_json_and_math4u_context(self):
        self.assertIn("NSW HSC Mathematics Extension 2", QUESTION_GENERATION_PROMPT)
        self.assertIn('"Question_Type"', QUESTION_GENERATION_PROMPT)
        self.assertIn('"Multiple Choices"', QUESTION_GENERATION_PROMPT)
        self.assertIn('"Answer"', QUESTION_GENERATION_PROMPT)
        self.assertIn('"Marks"', QUESTION_GENERATION_PROMPT)

    def test_feedback_prompt_is_student_friendly_and_preserves_answers(self):
        prompt = build_answer_feedback_prompt(
            "Proof Question",
            "Assume n is even.",
            "Let n = 2k, where k is an integer.",
        )
        self.assertIn("NSW HSC Mathematics Extension 2", prompt)
        self.assertIn("Assume n is even.", prompt)
        self.assertIn("Let n = 2k", prompt)
        self.assertIn("first important issue", prompt)
        self.assertNotIn("Physics tutor", prompt)

    def test_flash_card_prompt_is_math_specific_and_preserves_answer(self):
        prompt = build_flash_card_prompt(
            "Use integration by parts with u = x and dv = e^x dx."
        )
        self.assertIn("NSW HSC Mathematics Extension 2", prompt)
        self.assertIn("integration by parts", prompt)
        self.assertIn("Theorem, identity, or method", prompt)
        self.assertNotIn("physics law", prompt.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
```

Expected: `ERROR` because `src.math4u_prompts` does not exist.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_math4u_prompts.py
git commit -m "test: define Math 4U prompt contracts"
```

### Task 2: Implement the Central Math 4U Prompt Module

**Files:**
- Create: `src/math4u_prompts.py`
- Test: `tests/test_math4u_prompts.py`

- [ ] **Step 1: Create the complete prompt module**

```python
QUESTION_TYPES = [
    "Multiple Choice Question",
    "Calculation Question",
    "Proof Question",
    "Geometry Question",
    "Others",
]
DEFAULT_QUESTION_TYPE = "Others"


def normalize_question_type(value: str) -> str:
    cleaned = " ".join((value or "").strip().lower().replace("_", " ").split())
    aliases = {
        "multiple choice": "Multiple Choice Question",
        "multiple choice question": "Multiple Choice Question",
        "multiplechoice": "Multiple Choice Question",
        "mcq": "Multiple Choice Question",
        "calculation": "Calculation Question",
        "calculation question": "Calculation Question",
        "numerical": "Calculation Question",
        "proof": "Proof Question",
        "proof question": "Proof Question",
        "prove": "Proof Question",
        "show": "Proof Question",
        "show that": "Proof Question",
        "verify": "Proof Question",
        "geometry": "Geometry Question",
        "geometry question": "Geometry Question",
        "geometric": "Geometry Question",
        "other": "Others",
        "others": "Others",
        "other question": "Others",
        "other questions": "Others",
        "short answer": "Others",
        "short-answer": "Others",
        "essay": "Others",
        "extended response": "Others",
        "experimental": "Others",
        "experimental questions": "Others",
        "data analysis": "Others",
        "data analysis questions": "Others",
    }
    return aliases.get(cleaned, DEFAULT_QUESTION_TYPE)


QUESTION_GENERATION_PROMPT = """
You are an expert NSW HSC Mathematics Extension 2 teacher and assessment writer.
Study the supplied image carefully and create one high-quality HSC-style question
based only on the mathematical content that is visible.

Requirements:
- Use clear, syllabus-appropriate language and a realistic HSC command verb.
- Make the difficulty and mark value appropriate for Mathematics Extension 2.
- Include every condition needed to solve the question.
- Prefer exact values unless the question explicitly requests an approximation.
- Do not invent unreadable labels, values, or diagram features.
- If the image is too unclear to use reliably, say so instead of guessing.
- Classify the question using exactly one approved label:
  Multiple Choice Question, Calculation Question, Proof Question,
  Geometry Question, or Others.
- For a multiple-choice question, provide exactly four plausible options and
  identify the correct option. For every other type, use an empty options list
  and provide the mathematical answer.

Return valid JSON only, with exactly this structure:
{
  "Question": "Complete question text",
  "Question_Type": "One approved label",
  "Image_DataTable": "Python code needed to reproduce a useful visual, or an empty string",
  "Multiple Choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "Answer": "Correct option and/or concise mathematical answer",
  "Marks": 2
}
""".strip()


TEXT_ANSWER_PROMPT = """
You are an expert, supportive NSW HSC Mathematics Extension 2 tutor.
Read the complete question in the supplied image, then write a rigorous worked
solution that a student can learn from and use as a model HSC response.

Solution requirements:
1. Answer every part of the question and follow its command verb.
2. State relevant definitions, restrictions, domains, assumptions, or initial
   conditions before using them.
3. Show a logical sequence of justified steps. Do not skip the key step that
   earns the marks.
4. For proofs, state what is being proved, justify each implication, and finish
   with an explicit conclusion.
5. For geometry, name the geometric property or diagram relationship used.
6. Prefer exact values unless an approximation is requested, and include units
   where they are meaningful.
7. Check that the final result satisfies the question and any stated domain.
8. Match the detail to the likely mark value. Be clear and student-friendly,
   but avoid padding or unrelated theory.
9. If any essential text or diagram detail is unreadable, identify what is
   missing rather than inventing it.

Use `$...$` for inline mathematics and `$$...$$` for display mathematics.
Do not use `\\(...\\)` or `\\[...\\]`.

At the very end, add exactly:
###QUESTION_TYPE
<Multiple Choice Question, Calculation Question, Proof Question, Geometry Question, or Others>
""".strip()


CLASSIFY_QUESTION_TYPE_PROMPT = """
Classify the NSW HSC Mathematics Extension 2 question in the supplied image by
the primary response skill required for full marks.

Return exactly one of:
Multiple Choice Question
Calculation Question
Proof Question
Geometry Question
Others

Rules:
- Use Multiple Choice Question whenever selectable answer options are supplied.
- Use Proof Question when the main command is prove, show, verify, establish,
  demonstrate, or justify a general result.
- Use Geometry Question when the main reasoning depends on a diagram, locus,
  construction, or geometric relationship.
- Use Calculation Question when symbolic or numerical computation is the main task.
- Use Others only when none of the first four labels fits cleanly.
- If labels overlap, follow the explicit command verb and the dominant work
  needed for full marks.

Return only the exact label and nothing else.
""".strip()


GRAPH_ANSWER_PROMPT = """
You are an expert, supportive NSW HSC Mathematics Extension 2 tutor.
Read the supplied question carefully and solve it rigorously. Include a graph
only when it materially helps answer the question or explain the method.

Return plain text only, using these exact section headers:

###ANSWER
Give the concise final answer, including exact values, restrictions, and units
where relevant.

###PLOT_CODE
If a graph is genuinely useful, provide executable Python code that defines
`generate_plot()` and returns a Plotly or Matplotlib figure through a variable
named `fig`. Label axes, important points, intercepts, asymptotes, and relevant
domain boundaries. If no graph is useful, leave this section blank.

###EXPLANATION
Give clear, justified working suitable for NSW HSC Mathematics Extension 2.
Connect graph features to the algebra or geometry and do not invent unreadable
information from the image.

###OTHERS
Add only a brief check, alternative method, or common student trap that is
genuinely useful. Otherwise leave this section blank.

Do not return JSON and do not wrap the response in a Markdown code fence.
Use `$...$` for inline mathematics and `$$...$$` for display mathematics.
""".strip()


def build_answer_feedback_prompt(
    question_type: str,
    student_answer: str,
    teacher_answer: str,
) -> str:
    return f"""
You are a supportive NSW HSC Mathematics Extension 2 tutor. Compare the
student's response with the saved teacher solution. Judge the mathematics,
method, reasoning, notation, restrictions, and conclusion rather than checking
only whether the final value matches.

Question type: {question_type}

Student answer:
{student_answer}

Saved teacher answer:
{teacher_answer}

Give concise, encouraging feedback in exactly this structure:
- Result: Correct, Partly correct, or Needs work
- What was done well:
- First important issue:
- How to improve it:
- Improved answer:

For a proof, check logical completeness and the final conclusion. For a
calculation, identify the first incorrect or unsupported step. For geometry,
check that each claimed relationship is justified. If the student's method is
different but valid, accept it. Keep the improved answer short enough to be a
realistic student response.
""".strip()


def build_flash_card_prompt(saved_answer: str) -> str:
    return f"""
You are a concise NSW HSC Mathematics Extension 2 study coach. Create one
exam-focused flash card from the saved worked answer below.

Saved answer:
{saved_answer}

Choose one reusable theorem, identity, method, condition, or common trap. Do not
turn the entire worked solution into a card.

Format exactly:
### Front
A short recall or application question.

### Back
- Key idea:
- Theorem, identity, or method:
- Conditions / restrictions:
- When to use it:
- Common trap:

Use `$...$` for inline mathematics and `$$...$$` for display mathematics.
Keep the card compact, precise, and student-friendly. If a field does not apply,
write `Not applicable`.
""".strip()
```

- [ ] **Step 2: Run the prompt-contract tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
```

Expected: all 8 tests pass.

- [ ] **Step 3: Commit the central prompt module**

```powershell
git add src/math4u_prompts.py tests/test_math4u_prompts.py
git commit -m "feat: add Math 4U prompt templates"
```

### Task 3: Connect Supporting Modules to the Prompt Source

**Files:**
- Modify: `src/student_answers.py:1-5,65-105,108-114`
- Modify: `src/llm.py:1-20`
- Test: `tests/test_math4u_prompts.py`

- [ ] **Step 1: Add integration assertions to the tests**

Add these imports:

```python
from src import llm
from src.student_answers import (
    build_answer_feedback_prompt as exported_feedback_prompt,
    build_flash_card_prompt as exported_flash_card_prompt,
    parse_flash_card,
)
```

Add these tests:

```python
    def test_supporting_modules_export_the_central_prompts(self):
        self.assertIs(llm.prompt_template, QUESTION_GENERATION_PROMPT)
        self.assertIs(exported_feedback_prompt, build_answer_feedback_prompt)
        self.assertIs(exported_flash_card_prompt, build_flash_card_prompt)

    def test_flash_card_fallback_is_math4u_specific(self):
        card = parse_flash_card("unstructured response")
        self.assertEqual(card["front"], "Review this key HSC Math 4U idea.")
```

- [ ] **Step 2: Run the tests to verify the integration assertions fail**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
```

Expected: failures because `src.llm` and `src.student_answers` still define their own prompt text and the fallback still says Physics.

- [ ] **Step 3: Replace duplicated builders in `src/student_answers.py`**

Add after the standard-library imports:

```python
from src.math4u_prompts import (
    build_answer_feedback_prompt,
    build_flash_card_prompt,
)
```

Delete the local `build_answer_feedback_prompt` and `build_flash_card_prompt` function definitions. Change the `parse_flash_card` fallback to:

```python
    return {
        "front": front_match.group(1).strip() if front_match else "Review this key HSC Math 4U idea.",
        "back": back_match.group(1).strip() if back_match else (text or "").strip(),
    }
```

- [ ] **Step 4: Replace the question-generation prompt in `src/llm.py`**

Add:

```python
from src.math4u_prompts import QUESTION_GENERATION_PROMPT
```

Replace the local triple-quoted prompt with:

```python
prompt_template = QUESTION_GENERATION_PROMPT
```

- [ ] **Step 5: Run the tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
```

Expected: all 10 tests pass.

- [ ] **Step 6: Commit supporting-module integration**

```powershell
git add src/student_answers.py src/llm.py tests/test_math4u_prompts.py
git commit -m "refactor: centralize Math 4U prompts"
```

### Task 4: Integrate Prompts and Classification in the Streamlit App

**Files:**
- Modify: `app.py:135-239,547-583,1093-1128,1568-1588,1760-1779`
- Test: `tests/test_math4u_prompts.py`

- [ ] **Step 1: Add a source-level app integration test**

Add imports:

```python
from pathlib import Path
```

Add:

```python
    def test_app_uses_central_prompts_and_new_multiple_choice_label(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("from src.math4u_prompts import (", app_source)
        self.assertIn("GRAPH_ANSWER_PROMPT", app_source)
        self.assertIn(
            'current_question_type == "Multiple Choice Question"',
            app_source,
        )
        self.assertNotIn('current_question_type == "Multiple choice"', app_source)
        self.assertNotIn("You are a top HSC teacher.", app_source)
```

- [ ] **Step 2: Run the tests to verify the app integration test fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
```

Expected: one failure because `app.py` still contains inline prompts and the legacy multiple-choice comparison.

- [ ] **Step 3: Import the central prompt contract in `app.py`**

Add with the other `src` imports:

```python
from src.math4u_prompts import (
    CLASSIFY_QUESTION_TYPE_PROMPT,
    DEFAULT_QUESTION_TYPE,
    GRAPH_ANSWER_PROMPT,
    QUESTION_TYPES,
    TEXT_ANSWER_PROMPT,
    normalize_question_type,
)
```

Consolidate the `src.student_answers` import so it directly includes:

```python
from src.student_answers import (
    append_answer_log,
    build_answer_feedback_prompt,
    build_answer_summary,
    build_flash_card_prompt,
    canonical_question_cache_key,
    estimate_study_panel_height,
    latest_answers_by_key,
    parse_flash_card,
    question_type_course_for_cache_key,
    read_json_list,
    study_markdown_to_html,
)
```

Delete the `try` / `except ImportError` fallback block that duplicates flash-card parsing and rendering helpers.

- [ ] **Step 4: Remove legacy question-type and prompt definitions**

Delete the local `QUESTION_TYPES`, `DEFAULT_QUESTION_TYPE`, and `normalize_question_type` definitions. Keep:

```python
QUESTION_TYPE_FILTERS = ["All types"] + QUESTION_TYPES
```

Delete the local `TEXT_ANSWER_PROMPT` and `CLASSIFY_QUESTION_TYPE_PROMPT` triple-quoted definitions because they are now imported.

- [ ] **Step 5: Connect the graph and multiple-choice behaviours**

Replace the inline graph prompt assignment with:

```python
                    prompt_answer = GRAPH_ANSWER_PROMPT
```

Change both multiple-choice conditionals to use:

```python
current_question_type == "Multiple Choice Question"
```

and:

```python
current_question_type != "Multiple Choice Question"
```

- [ ] **Step 6: Run prompt tests and compile the application**

Run:

```powershell
python -m unittest discover -s tests -p "test_math4u_prompts.py" -v
python -m compileall app.py src tests
```

Expected: all 11 tests pass and compilation completes without syntax errors.

- [ ] **Step 7: Commit app integration**

```powershell
git add app.py tests/test_math4u_prompts.py
git commit -m "feat: use Math 4U prompts in tutor app"
```

### Task 5: Final Verification

**Files:**
- Verify: `app.py`
- Verify: `src/math4u_prompts.py`
- Verify: `src/llm.py`
- Verify: `src/student_answers.py`
- Verify: `tests/test_math4u_prompts.py`

- [ ] **Step 1: Run the complete local verification suite**

```powershell
python -m unittest discover -s tests -v
python -m compileall app.py src tests
git diff --check
```

Expected: every test passes, Python compilation succeeds, and `git diff --check` reports no whitespace errors.

- [ ] **Step 2: Confirm stale Physics language is absent from non-video prompts**

Run:

```powershell
rg -n -i "Physics tutor|physics law|HSC Physics" app.py src/math4u_prompts.py src/llm.py src/student_answers.py
```

Expected: matches may remain only inside the explicitly excluded video recommendation prompt in `app.py`; there must be no match in `src/math4u_prompts.py`, `src/llm.py`, or `src/student_answers.py`.

- [ ] **Step 3: Inspect the final change set**

```powershell
git status --short
git diff --stat HEAD~3..HEAD
git log -4 --oneline
```

Expected: only the planned prompt, integration, test, and documentation files changed; the three implementation commits appear after the design/plan documentation commits.
