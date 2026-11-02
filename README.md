# 🎯 AI Resume vs Job Description Matcher

An AI-powered ATS (Applicant Tracking System) tool that evaluates resumes against job descriptions using local sentence-level vector embeddings and generates structured recruitment feedback via an LLM.

---

## 🚀 Key Features

- **Granular Semantic Matching:** Rather than comparing texts as single dense blocks, it breaks down requirements line-by-line and computes cosine similarity against candidate experience using Sentence-Transformers.
- **Identifies Gaps & Strengths:** Classifies requirements into matched skills and missing critical areas using a deterministic similarity threshold.
- **ATS Feedback via LLM:** Utilizes Groq-hosted open LLMs to generate structured JSON feedback containing recruiter summaries, key strengths, critical gaps, and tailored resume bullet points.
- **Multi-Format Input:** Supports PDF parsing as well as raw text input.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Vector Math:** Scikit-Learn (Cosine Similarity), NumPy
- **LLM Engine:** Groq API (`openai/gpt-oss-120b`)
- **Data Validation & Parsing:** Pydantic, PyPDF

---

## ⚙️ Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MuhammadHaris1122/resume-matcher.git](https://github.com/MuhammadHaris1122/resume-matcher.git)
   cd resume-matcher
## live App Hosted On Streamlit
https://resume-matcher-bxfwarfypswfmhs8gfublr.streamlit.app/
