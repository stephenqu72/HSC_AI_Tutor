# Math 4U Prompt Templates Design

## Goal

Rewrite the application's non-video LLM prompts for NSW HSC Mathematics Extension 2 (Math 4U). The prompts must produce mathematically rigorous, exam-relevant responses while remaining clear, encouraging, and easy for students to follow.

## Scope

The rewrite covers:

- HSC-style question generation
- Worked text answers
- Question classification
- Graph-assisted answers
- Student-answer feedback
- Flash-card generation

YouTube video recommendation prompts are explicitly excluded.

## Structure

Prompt text will be centralised in `src/math4u_prompts.py`. The application and supporting modules will import constants or builder functions from this module instead of maintaining overlapping inline prompts.

The existing response contracts will remain stable:

- Question generation continues to return the expected JSON object.
- Worked answers continue to end with `###QUESTION_TYPE`.
- Graph answers continue to use `###ANSWER`, `###PLOT_CODE`, `###EXPLANATION`, and `###OTHERS`.
- Flash cards continue to use `### Front` and `### Back`.

## Classification Contract

Every classification prompt and parser will use exactly these labels:

1. `Multiple Choice Question`
2. `Calculation Question`
3. `Proof Question`
4. `Geometry Question`
5. `Others`

Normalization will accept common legacy aliases and map them to the new labels. The multiple-choice answer widget will compare against `Multiple Choice Question`, preserving its current radio-button behaviour.

The categories describe the dominant response skill:

- `Multiple Choice Question`: the student selects from supplied options.
- `Calculation Question`: the main task is symbolic or numerical computation.
- `Proof Question`: the main task is proving, showing, verifying, or establishing a result.
- `Geometry Question`: the main task relies on a geometric diagram, locus, construction, or spatial argument.
- `Others`: questions that do not fit one of the four categories cleanly.

When categories overlap, the classifier will prefer the explicit command verb and the primary work required for full marks.

## Prompt Behaviour

All prompts will identify the subject as NSW HSC Mathematics Extension 2 rather than Physics or generic HSC teaching.

Worked solutions will:

- Read the image carefully and avoid inventing missing information.
- State relevant restrictions, domains, assumptions, and definitions.
- Show a logical sequence of justified steps suitable for HSC marking.
- Prefer exact values unless an approximation is requested.
- Use correct mathematical notation and consistent variables.
- Check the final result against the question.
- Match the response depth to the marks and command verb.
- Explain difficult transitions in student-friendly language without padding.
- Format LaTeX with `$...$` and `$$...$$` only.

Proof responses will clearly identify what is being proved, justify each implication, and finish with an explicit conclusion. Calculation responses will retain meaningful working rather than jumping to an unsupported answer. Geometry responses will connect algebraic steps to the diagram or geometric property being used.

Graph prompts will generate plotting code only when a graph materially helps answer or explain the question. Generated code must keep the existing `generate_plot()` interface and return a figure through `fig`.

Feedback prompts will assess mathematical method and reasoning, not merely compare final answers. Feedback will identify what was correct, the first important issue, how to fix it, and a concise improved response. The tone will be encouraging and specific.

Flash cards will focus on one reusable theorem, method, identity, condition, or common trap from the saved answer. They will avoid Physics-oriented wording such as laws, physical constants, or figures unless such wording is genuinely relevant to a mechanics question.

## Error Handling

Prompts will instruct the model to say when an image is unreadable or information is missing rather than fabricate a solution. Parsers will continue to fall back to `Others` when classification output is absent or invalid.

## Testing

Prompt contract tests will verify:

- All five exact classification labels are present.
- Legacy labels normalize correctly.
- Invalid classifications fall back to `Others`.
- Required response headers remain unchanged.
- Multiple-choice UI logic uses the new label.
- Feedback and flash-card builders preserve supplied content.
- Non-video prompts contain Math 4U / Mathematics Extension 2 context and no stale Physics tutor wording.
- Video recommendation prompt text remains unchanged.

## Non-Goals

- Changing the visual design or user workflow
- Rewriting saved generated answers
- Adding live syllabus lookup or external references
- Changing model providers or generation settings
- Modifying video recommendation behaviour
