from __future__ import annotations

import json
import os
import re
from io import BytesIO

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


MODEL = "gemini-3.5-flash"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_TEXT_CHARS = 30_000

st.set_page_config(
    page_title="Entrance Exam Hub & AI Tutor",
    page_icon="📚",
    layout="wide",
)

st.title("🎓 Entrance Exam Hub & AI Tutor")
st.caption("Upload a question paper and turn its questions into an interactive quiz with worked explanations.")

with st.sidebar:
    st.title("⚙️ Configuration")
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = st.text_input(
        "Google Gemini API key",
        value=secret_key,
        type="password",
        help="For Streamlit Cloud, set GEMINI_API_KEY under app Settings → Secrets.",
    )
    st.markdown("---")
    st.markdown("### 🏛️ Portal Ownership")
    st.markdown("**Owner / Creator:** NilanjanC")


def extract_text(upload) -> str:
    """Extract text from supported, text-based PDF and Word documents."""
    payload = upload.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The file is over 15 MB. Please upload a smaller paper.")

    suffix = upload.name.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        reader = PdfReader(BytesIO(payload))
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        text = "\n".join(chunks)
    elif suffix == "docx":
        document = Document(BytesIO(payload))
        text = "\n".join(p.text for p in document.paragraphs)
        for table in document.tables:
            for row in table.rows:
                text += "\n" + " | ".join(cell.text for cell in row.cells)
    else:
        raise ValueError("Please upload a PDF or Word (.docx) file.")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text and suffix != "pdf":
        raise ValueError("No text was found in this Word document.")
    return text[:MAX_TEXT_CHARS]


def parse_model_json(raw_text: str) -> dict:
    """Read the first complete JSON value even if the model adds trailing text."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

    decoder = json.JSONDecoder()
    starts = [position for position, char in enumerate(text) if char in "{["]
    for start in starts:
        try:
            parsed, _end = decoder.raw_decode(text, start)
            if isinstance(parsed, list):
                return {"questions": parsed}
            if isinstance(parsed, dict):
                if isinstance(parsed.get("questions"), list):
                    return parsed
                if "question" in parsed:
                    return {"questions": [parsed]}
        except json.JSONDecodeError:
            continue
    raise ValueError("Gemini's response did not contain a complete JSON quiz. Please try again.")


def build_quiz(document_text: str, count: int, subject: str, key: str, pdf_bytes: bytes | None = None) -> list[dict]:
    client = genai.Client(api_key=key)
    prompt = f"""Create exactly {count} quiz questions based only on the supplied exam paper.
Subject/exam context: {subject or 'Infer from the paper'}.
For each item, preserve the meaning of an original question where possible. Turn it into a clear multiple-choice question with exactly 4 plausible options. Give one correct option, a concise step-by-step explanation, and the source question number if visible (otherwise an empty string).
Do not invent facts from the paper. If the document has fewer than {count} usable questions, return only the usable ones. Return valid JSON with this shape:
{{"questions":[{{"question":"...","options":["...","...","...","..."],"answer":"exact option text","explanation":"...","source_number":"..."}}]}}

Exam paper text:
{document_text}
"""
    contents = [prompt]
    if pdf_bytes:
        # Gemini can read a PDF visually, including scanned pages and diagrams.
        contents.insert(0, types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )
    data = parse_model_json(response.text or "")
    questions = data.get("questions", [])
    valid = []
    for q in questions:
        options = q.get("options", [])
        answer = q.get("answer", "")
        if (
            isinstance(q.get("question"), str)
            and len(options) == 4
            and answer in options
            and isinstance(q.get("explanation"), str)
        ):
            valid.append(q)
    if not valid:
        returned_count = len(questions) if isinstance(questions, list) else 0
        raise ValueError(
            f"The AI returned {returned_count} question(s), but none matched the required quiz format. "
            "Try again, or use a clearer paper with visible question text."
        )
    return valid


paper_tab, solver_tab, support_tab = st.tabs(
    ["📄 Question Papers Hub", "🤖 AI Question & Quiz Solver", "❤️ Support"]
)

with paper_tab:
    st.header("Exam practice")
    exams = {
        "NEET (Medical)": "National Eligibility cum Entrance Test for undergraduate medical programs.",
        "JEE Main (Engineering)": "Joint Entrance Examination for admission to NITs, IIITs, and other participating institutes.",
        "JEE Advanced": "Entrance examination for admission to the Indian Institutes of Technology.",
        "IAT - IISER Aptitude Test": "Entrance test for undergraduate science programmes at IISERs.",
        "NEST": "Entrance test for the integrated MSc programmes at NISER and UM-DAE CEBS.",
        "CUET-UG": "Common University Entrance Test for participating universities.",
        "ICAR/AIEEA-UG": "Entrance pathway for undergraduate agriculture programmes.",
        "BITSAT": "Birla Institute of Technology and Science Admission Test.",
    }
    selected_exam = st.selectbox("Choose an examination", list(exams))
    st.info(f"**{selected_exam}:** {exams[selected_exam]}")
    st.markdown("Use the **AI Question & Quiz Solver** tab to upload your own paper.")
    st.caption("Sample paper download links can be added here when verified source URLs are available.")

with solver_tab:
    st.header("Turn a question paper into a quiz")
    st.write("Upload a text-based PDF or Word document. The app extracts its questions and prepares answer explanations.")
    uploaded_file = st.file_uploader(
        "Question paper", type=["pdf", "docx"], help="Maximum 15 MB. Scanned image-only PDFs need OCR and are not supported yet."
    )
    subject = st.text_input("Subject or exam (optional)", placeholder="e.g. NEET Chemistry")
    question_count = st.slider("Questions to include", min_value=1, max_value=20, value=10)

    extracted = ""
    if uploaded_file:
        try:
            extracted = extract_text(uploaded_file)
            if extracted:
                st.success(f"Read **{len(extracted):,} characters** from `{uploaded_file.name}`.")
                with st.expander("Preview extracted text"):
                    st.text(extracted[:2_000] + (" …" if len(extracted) > 2_000 else ""))
            else:
                st.info("No selectable text was extracted. Gemini will try to read the PDF pages as images.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    if st.button("Create my quiz", type="primary", disabled=uploaded_file is None, use_container_width=True):
        # Remove results from an earlier paper before attempting this one.
        for key in ["exam_quiz", "exam_answers", "exam_submitted", "exam_source_name"]:
            st.session_state.pop(key, None)
        if not api_key.strip():
            st.warning("Add your Gemini API key in the sidebar, or configure GEMINI_API_KEY in Streamlit Secrets.")
        else:
            try:
                with st.spinner("Reading the paper and preparing your quiz…"):
                    pdf_bytes = uploaded_file.getvalue() if uploaded_file.name.lower().endswith(".pdf") else None
                    st.session_state.exam_quiz = build_quiz(
                        extracted, question_count, subject.strip(), api_key.strip(), pdf_bytes
                    )
                st.session_state.exam_answers = {}
                st.session_state.exam_submitted = set()
                st.session_state.exam_source_name = uploaded_file.name
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")

    quiz = st.session_state.get("exam_quiz", [])
    if quiz:
        st.divider()
        st.subheader(f"Your quiz · {st.session_state.get('exam_source_name', 'Question paper')}")
        for index, item in enumerate(quiz):
            label = f"Question {index + 1}"
            if item.get("source_number"):
                label += f" · Paper question {item['source_number']}"
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(item["question"])
                selected = st.radio(
                    "Choose one answer",
                    item["options"],
                    key=f"exam_answer_{index}",
                    index=None,
                    label_visibility="collapsed",
                )
                if st.button("Check answer", key=f"exam_check_{index}", disabled=selected is None):
                    st.session_state.exam_answers[index] = selected
                    st.session_state.exam_submitted.add(index)
                if index in st.session_state.get("exam_submitted", set()):
                    chosen = st.session_state.exam_answers[index]
                    if chosen == item["answer"]:
                        st.success("Correct!")
                    else:
                        st.error(f"Not quite. Correct answer: {item['answer']}")
                    st.info(f"**Explanation:** {item['explanation']}")

        answered = st.session_state.get("exam_submitted", set())
        if answered:
            score = sum(st.session_state.exam_answers[i] == quiz[i]["answer"] for i in answered)
            st.metric("Score so far", f"{score} / {len(answered)}")
        if st.button("Start over"):
            for key in ["exam_quiz", "exam_answers", "exam_submitted", "exam_source_name"]:
                st.session_state.pop(key, None)
            st.rerun()

with support_tab:
    st.header("Support this platform")
    st.write("A payment button can be added here after you create one in Razorpay.")
    st.caption("The original file contained a placeholder payment-button ID, so it could not accept donations as written.")

st.divider()
st.caption("© 2026 Entrance Exam Hub · Built with Streamlit and Gemini · Owner: NilanjanC")
