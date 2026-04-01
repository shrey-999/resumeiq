# Demo Video link : https://www.loom.com/share/c00431b24b1c4f53b7b6da95a7e639e3
# AI Resume-Job Matcher

A professional and efficient tool to evaluate candidate resumes against job descriptions. This application processes multiple resumes, ranks them by relevance, and provides actionable insights based on rule-based heuristics and optional LLM enhancements.

## Features

- **Multi-Resume Support:** Upload multiple PDF resumes simultaneously to process and rank them in a single batch.
- **Rule-Based Evaluation:** Provides deterministic strengths and improvement suggestions without relying on external APIs, ensuring it functions 100% locally.
- **LLM Enhanced Insights (Optional):** If a Gemini API key is provided, the application enhances the feedback for the top candidate.
- **Semantic Matching:** Utilizes `SentenceTransformers` for dense vector embeddings and similarity computations across distinct resume sections (Skills, Experience, Projects).
- **Keyword Extraction:** Leverages `spaCy` NLP to intelligently extract required keywords from the job description and identify gaps in the resume.

## Architecture

1. **Document Parsing:** PyMuPDF (`fitz`) extracts text from uploaded PDF resumes.
2. **Heuristic Validation:** Ensures that uploaded documents contain standard resume headers and sufficient content.
3. **Section Classification:** Text is intelligently split into Skills, Experience, Projects, and Education modules.
4. **Vector Embeddings:** Uses the `all-MiniLM-L6-v2` transformer model to create embeddings for the Job Description and individual resume sections.
5. **Similarity Scoring:** Computes cosine similarity between section embeddings, applying weighted calculations to produce an overall match score.
6. **Insight Generation:** Evaluates scores and keyword overlaps to generate rule-based Strengths and Improvement Suggestions.
7. **Ranking:** Ranks multiple valid resumes by overall match score.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd resumeiq
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On your first run, the app will automatically download the `en_core_web_sm` spaCy model if it's missing.*

3. **Configure Environment Variables (Optional):**
   To enable enhanced LLM insights for the top match, create a `.env` file from the provided example:
   ```bash
   cp .env.example .env
   ```
   Add your API key inside `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
The application will be available in your browser at `http://localhost:8501`.
