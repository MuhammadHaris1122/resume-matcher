import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)

st.set_page_config(page_title="AI Resume vs JD Matcher", page_icon="🎯", layout="wide")

# Cache model so it loads only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# Core Matching Engine
def compute_matching(resume_text, jd_text, threshold=50.0):
    resume_lines = [line.strip("- *• \t\n") for line in resume_text.strip().split("\n") if len(line.strip()) > 5]
    jd_lines = [line.strip("- *• \t\n") for line in jd_text.strip().split("\n") if len(line.strip()) > 5]

    if not resume_lines or not jd_lines:
        return None, None, None, None

    resume_embeddings = model.encode(resume_lines, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_lines, convert_to_tensor=True)

    sim_matrix = cos_sim(jd_embeddings, resume_embeddings).cpu().numpy()

    results = []
    for i, jd_req in enumerate(jd_lines):
        best_idx = int(np.argmax(sim_matrix[i]))
        score = round(float(sim_matrix[i][best_idx]) * 100, 2)
        results.append({
            "Requirement": jd_req,
            "Best Resume Match": resume_lines[best_idx],
            "Match Score (%)": score
        })

    df = pd.DataFrame(results)
    overall_score = round(df["Match Score (%)"].mean(), 2)
    matched = df[df["Match Score (%)"] >= threshold]
    gaps = df[df["Match Score (%)"] < threshold]

    return overall_score, matched, gaps, df

# LLM Feedback Generator
def generate_llm_feedback(overall_score, matched_df, gaps_df, resume_text, api_key):
    client = Groq(api_key=api_key)

    matched_list = "\n".join([f"- {row['Requirement']} ({row['Match Score (%)']}%)" for _, row in matched_df.iterrows()])
    gaps_list = "\n".join([f"- {row['Requirement']} ({row['Match Score (%)']}%)" for _, row in gaps_df.iterrows()])

    system_prompt = """
    You are an expert technical recruiter and ATS specialist.
    Analyze the match evaluation and return strictly valid JSON matching this schema:
    {
      "summary": "1-2 sentence overall evaluation",
      "key_strengths": ["list of identified strengths"],
      "critical_gaps": ["list of critical missing areas to address"],
      "actionable_resume_bullets": ["2-3 specific, tailored bullet points candidate can add to resume to bridge the gaps honestly"]
    }
    Do not use markdown backticks. Return pure JSON only.
    """

    user_prompt = f"""
    Overall Score: {overall_score}%

    Matched:
    {matched_list}

    Gaps:
    {gaps_list}

    Candidate Resume Text:
    {resume_text}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ----------------- UI Layout -----------------
st.title("🎯 AI Resume vs Job Description Matcher")
st.caption("Powered by Local Sentence-Transformers & Groq LLM")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Candidate Resume")
    uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    resume_manual = st.text_area("Or Paste Resume Text directly:", height=200)

with col2:
    st.subheader("📋 Job Description")
    jd_input = st.text_area("Paste the Target Job Description here:", height=290)

if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    # Determine resume text source
    resume_content = ""
    if uploaded_pdf:
        resume_content = extract_text_from_pdf(uploaded_pdf)
    elif resume_manual.strip():
        resume_content = resume_manual.strip()

    if not resume_content or not jd_input.strip():
        st.error("Please provide both a Resume and a Job Description.")
    elif not groq_api_key:
        st.error("GROQ_API_KEY is missing. Check your .env file.")
    else:
        with st.spinner("Calculating semantic match vectors and generating ATS report..."):
            overall_score, matched_df, gaps_df, full_df = compute_matching(resume_content, jd_input)
            feedback = generate_llm_feedback(overall_score, matched_df, gaps_df, resume_content, groq_api_key)

        st.divider()

        # Score Metric Banner
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Overall ATS Match Score", f"{overall_score}%")
        mcol2.metric("Matched Requirements", f"{len(matched_df)}")
        mcol3.metric("Identified Gaps", f"{len(gaps_df)}")

        # AI Summary
        st.info(f"**Recruiter Assessment:** {feedback.get('summary')}")

        # Strengths & Gaps Columns
        scol1, scol2 = st.columns(2)
        with scol1:
            st.success("### ✅ Key Strengths")
            for strength in feedback.get("key_strengths", []):
                st.markdown(f"- {strength}")

        with scol2:
            st.error("### ⚠️ Critical Gaps")
            for gap in feedback.get("critical_gaps", []):
                st.markdown(f"- {gap}")

        # Actionable Bullets
        st.subheader("💡 Tailored Resume Action Points")
        for bullet in feedback.get("actionable_resume_bullets", []):
            st.markdown(f"👉 {bullet}")

        # Granular Breakdown Table
        with st.expander("🔍 View Detailed Sentence-by-Sentence Score Breakdown"):
            st.dataframe(full_df, use_container_width=True)
