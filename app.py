import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# load spacy models once so it doesn't freeze the app
@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

# load embedding model (used for semantic matching)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# --- Parsing & Extraction ---

def read_pdf(file_bytes):
    # simple pdf reader using pymupdf
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception:
        return ""

def is_valid_resume(text):
    if len(text) < 100:
        return False
        
    text_lower = text.lower()
    # basic check to see if this actually looks like a resume
    keywords = ["experience", "skills", "education", "projects", "work", "employment", "professional"]
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches >= 1

def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "projects": "",
        "education": ""
    }
    
    current_sec = "experience" 
    
    # trying to catch both standard and academic headers
    triggers = {
        "skills": ['SKILLS', 'TECHNICAL SKILLS', 'CORE COMPETENCIES', 'INTEREST AREAS', 'RESEARCH INTERESTS', 'AREAS OF EXPERTISE', 'EXPERTISE', 'INTERESTS', 'AREAS'],
        "experience": ['EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT HISTORY'],
        "projects": ['PROJECTS', 'PERSONAL PROJECTS', 'ACADEMIC PROJECTS', 'PROJECT', 'RESEARCH PROJECTS', 'CONSULTANCY PROJECTS', 'PUBLICATIONS', 'RESEARCH TOPICS'],
        "education": ['EDUCATION', 'ACADEMIC BACKGROUND']
    }
    
    for line in text.split('\n'):
        clean_line = re.sub(r'[^A-Z\s]', '', line.strip().upper()).strip()
        
        # if the line is short, it might be a header
        if 0 < len(clean_line) < 40:
            found_header = False
            for sec_name, sec_triggers in triggers.items():
                if any(clean_line == t or clean_line.startswith(t) for t in sec_triggers):
                    current_sec = sec_name
                    found_header = True
                    break
            
            # skip adding the header itself to the section text
            if found_header:
                continue 
                
        sections[current_sec] += line + "\n"
        
    return {k: v.strip() for k, v in sections.items()}


# --- NLP & Matching Logic ---

def clean_kw(chunk_text):
    # strips out boilerplate words from jd
    clean_text = chunk_text.strip().lower()
    prefixes = ["experience with", "knowledge of", "familiarity with", "understanding of", "ability to"]
    suffixes = ["methodologies", "techniques", "frameworks", "tools", "systems"]
    
    for p in prefixes:
        if clean_text.startswith(p):
            clean_text = clean_text[len(p):].strip()
    for s in suffixes:
        if clean_text.endswith(s):
            clean_text = clean_text[:-len(s)].strip()
            
    return clean_text

def extract_keywords(text, nlp):
    # pulls out noun phrases from the job description
    doc = nlp(text)
    raw_kws = []
    
    for chunk in doc.noun_chunks:
        cln = clean_kw(chunk.text)
        if 3 <= len(cln) <= 40 and "\n" not in cln:
            raw_kws.append(cln)
            
    stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
    noise = {
        "experience", "responsibilities", "skills", "knowledge", "ability",
        "work", "years", "team", "requirements", "project", "role",
        "candidate", "understanding", "opportunity", "description",
        "job", "looking", "environment", "position", "company", "business",
        "strong", "required", "familiarity", "expertise", "professor",
        "associate professor", "a professor", "students", "researcher",
        "manager", "director"
    }
    
    freqs = {}
    for kw in raw_kws:
        tokens = kw.split()
        # make sure it's not just a bunch of stop words or generic terms
        if not all(t in stopwords for t in tokens) and not any(n in kw for n in noise):
            if not kw.isdigit() and len(kw) > 2:
                freqs[kw] = freqs.get(kw, 0) + 1
                
    # rank by how often they appear and prefer multi-word phrases (bigrams > unigrams)
    ranked = sorted(freqs.keys(), key=lambda x: (freqs[x], len(x.split())), reverse=True)
    return ranked[:8]

def normalize(text):
    # lowercase and remove punctuation
    text = text.lower()
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())

def stem_word(word):
    # simple python stemmer without needing nltk
    w = word.lower().strip()
    if len(w) <= 3:
        return w
        
    for suffix in ["ing", "ed", "es", "s"]:
        if w.endswith(suffix):
            stemmed = w[:-len(suffix)]
            if len(stemmed) >= 3: 
                return stemmed
    return w

def get_embedding(texts, model):
    if isinstance(texts, str):
        if not texts.strip():
            return model.encode(" ")
    return model.encode(texts)

def cos_sim(vec1, vec2):
    # compare how similar vectors are using cosine similarity
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    sim = dot / (norm1 * norm2)
    # scale from [-1, 1] to [0, 100]
    score = ((sim + 1) / 2) * 100
    return max(0.0, min(100.0, score))

def match_skills(resume_text, req_kws, encoder, nlp):
    norm_res = normalize(resume_text)
    
    doc = nlp(resume_text)
    # grabbing actual sentences for the semantic fallback
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    sent_vecs = None
    
    matched, missing = [], []
    partial_hits, semantic_hits = [], []
    
    for kw in req_kws:
        norm_k = normalize(kw)
        stem_k = stem_word(norm_k)
        
        # 1. exact or normalized string match
        if norm_k in norm_res or stem_k in norm_res:
            matched.append(kw)
            continue
            
        # 2. partial match (e.g. tracking both "project" and "manage" in the same sentence)
        tokens = norm_k.split()
        stems = [stem_word(t) for t in tokens]
        
        found_partial = False
        if len(stems) > 1: 
            for sentence in sents:
                norm_sent = normalize(sentence)
                if all(s in norm_sent for s in stems):
                    found_partial = True
                    break
        
        if found_partial:
            matched.append(kw)
            partial_hits.append(kw)
            continue
            
        # 3. semantic boost (using embeddings as a last resort)
        if sent_vecs is None and sents:
            sent_vecs = get_embedding(sents, encoder)
            
        found_semantic = False
        if sents:
            kw_vec = get_embedding(kw, encoder)
            for i in range(len(sents)):
                sim = cos_sim(kw_vec, sent_vecs[i])
                if sim >= 60.0:  # 60 feels balanced for short keyword vs full sentence
                    found_semantic = True
                    break
                    
        if found_semantic:
            matched.append(kw)
            semantic_hits.append(kw)
        else:
            missing.append(kw)
            
    return matched, missing, partial_hits, semantic_hits


# --- Scoring & Feedback ---

def compute_score(sec_scores, kw_match_pct):
    scaled = sec_scores.copy()
    
    # if they barely match any keywords, heavily downscale their section scores
    if kw_match_pct < 30.0:
        ratio = max(0.1, kw_match_pct / 100.0)
        for s in scaled:
            scaled[s] *= ratio

    weights = {"skills": 0.40, "experience": 0.40, "projects": 0.20}
    emb_total = 0
    
    for sec, w in weights.items():
        val = scaled.get(sec, 0.0)
        emb_total += val * w
            
    final = (emb_total * 0.60) + (kw_match_pct * 0.40)
    
    # penalty for missing too many critical keywords
    if kw_match_pct < 40.0:
        final -= 10.0
    elif kw_match_pct < 60.0:
        final -= 5.0
        
    # hardcap to ensure perfect keywords still gets a decent score no matter what
    if kw_match_pct >= 95.0:
        final = max(76.0, final)
        
    return max(0.0, min(100.0, final)), scaled

def get_tips(missing_kws, scaled_scores):
    tips = []
    
    if missing_kws:
        targets = missing_kws[:2]
        impact = min(15, len(missing_kws) * 3)
        formatted_kws = " and ".join([f"'{k}'" for k in targets])
        tips.append(f"Adding {formatted_kws} could increase your score by ~{impact}-{impact+5}%.")
        
    if scaled_scores.get('projects', 0) < 40:
        tips.append("Improving your Projects section explicitly could increase your score by ~5-10%.")
        
    if scaled_scores.get('experience', 0) < 50:
        tips.append("Fleshing out your Experience metrics could increase your score by ~10-15%.")
        
    return tips

def rule_based_feedback(scores, missing):
    strengths, improvements = [], []
    
    # strengths
    if scores.get('skills', 0) > 70:
        strengths.append("Strong alignment in relevant technical/domain skills.")
    if scores.get('experience', 0) > 70:
        strengths.append("Strong professional experience relevant to the role.")
    if scores.get('projects', 0) > 70:
        strengths.append("Good project/initiative portfolio mapping to job requirements.")
    
    if scores.get('overall', 0) > 75:
        strengths.append("Overall excellent match for this position.")
    elif not strengths:
        strengths.append("Resume contains valid foundational sections for review.")

    # improvements
    if missing:
        top = missing[:3]
        improvements.append(f"Add missing explicit skills directly into your matrix (e.g., {', '.join(top)}).")
    
    improvements.append("Add quantified achievements (e.g., improved efficiency by 20%, reduced latency by 15%).")
    improvements.append("Mention explicitly the tools/frameworks used (Jira, specific softwares, environments) to support your experience.")
        
    return {"strengths": strengths, "improvements": improvements}

def analyze_with_gemini(scores, missing_kws, res_text, jd_text, rule_fb):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return rule_fb
        
    genai.configure(api_key=key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
You are an expert AI Resume Matcher and recruiter. 
You evaluated a candidate against a job description. 

Match Data:
- Overall Score: {scores['overall']:.1f}%
- Skills Match: {scores.get('skills', 0):.1f}%
- Experience Match: {scores.get('experience', 0):.1f}%
- Projects Match: {scores.get('projects', 0):.1f}%
Missing Core Keywords: {', '.join(missing_kws) if missing_kws else 'None'}

Using these metrics, list out specific bullet points. Do not hallucinate capabilities the candidate does not have.
Return precisely 2 sections titled exactly "STRENGTHS:" and "IMPROVEMENTS:".
"""
        res = model.generate_content(prompt)
        parts = res.text.split("IMPROVEMENTS:")
        str_raw = parts[0].replace("STRENGTHS:", "").strip()
        imp_raw = parts[1].strip() if len(parts) > 1 else "Analysis output parsing error."
        
        return {
            "strengths": [s.strip("- ") for s in str_raw.split("\n") if s.strip()],
            "improvements": [i.strip("- ") for i in imp_raw.split("\n") if i.strip()]
        }
    except Exception:
        return rule_fb

def process_file(file_obj, jd_text, jd_vec, final_kws, encoder, nlp):
    # setup default state so UI doesn't crash
    result = {
        "filename": file_obj.name,
        "score": 0.0,
        "scores_dict": {"overall": 0.0, "skills": 0.0, "experience": 0.0, "projects": 0.0},
        "section_scores": {"skills": 0.0, "experience": 0.0, "projects": 0.0},
        "matched_keywords": [],
        "missing_keywords": final_kws.copy() if final_kws else [],
        "partial_keywords": [],
        "semantic_keywords": [],
        "strengths": ["Resume contains foundational sections for review."],
        "improvements": ["Tailor the resume more clearly to the job description."],
        "score_guidance": [],
        "resume_text": "",
        "missing_sections": [],
        "error": None,
        "is_warning": False
    }
    
    try:
        res_text = read_pdf(file_obj.read())
        if not res_text:
            result["error"] = "No extractable text found or invalid PDF document."
            return result
            
        result["resume_text"] = res_text
        if not is_valid_resume(res_text):
            result["error"] = "Warning: Uploaded file may not be a valid resume."
            result["is_warning"] = True
            
        sections = extract_sections(res_text)
        sec_scores = {}
        missing_secs = []
        
        for sec in ["skills", "experience", "projects"]:
            content = sections.get(sec, "")
            
            # fallback for academic or weirdly structured resumes
            if sec == "skills" and len(content.strip()) < 10:
                content = res_text[:1500]
                
            if len(content.strip()) > 10:
                sec_vec = get_embedding(content, encoder)
                sec_scores[sec] = cos_sim(sec_vec, jd_vec)
            else:
                sec_scores[sec] = 0.0
                missing_secs.append(sec)
                
        matched, missing, partial, semantic = match_skills(res_text, final_kws, encoder, nlp)
        kw_pct = (len(matched) / len(final_kws) * 100) if final_kws else 100.0
                
        final_score, scaled = compute_score(sec_scores, kw_pct)
        
        scores_dict = {
            "overall": final_score,
            "skills": scaled.get('skills', 0.0),
            "experience": scaled.get('experience', 0.0),
            "projects": scaled.get('projects', 0.0)
        }
        
        fb = rule_based_feedback(scores_dict, missing)
        tips = get_tips(missing, scaled)
        
        result.update({
            "score": final_score,
            "scores_dict": scores_dict,
            "section_scores": scaled,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "partial_keywords": partial,
            "semantic_keywords": semantic,
            "strengths": fb["strengths"],
            "improvements": fb["improvements"],
            "score_guidance": tips,
            "missing_sections": missing_secs
        })
        
    except Exception as e:
        result["error"] = f"Could not analyze this document properly: {str(e)}"
        
    return result


# --- UI Application ---

def main():
    st.set_page_config(page_title="ResumeIQ Matcher", layout="wide")
    st.title("AI Resume-Job Matcher")
    st.markdown("Easily compare your resume against a job description. Computes intelligent vector similarity on categorized sections, and provides actionable insights.")
    
    if not os.path.exists(".env"):
        st.info("To enable LLM insights, create a `.env` file with `GEMINI_API_KEY=your_key_here`.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Resume Input")
        files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
        
    with col2:
        st.subheader("2. Job Description")
        jd_input = st.text_area("Paste Job Description here", height=200)
        
    if st.button("Analyze Alignment", type="primary"):
        if not files:
            st.warning("Please upload at least one PDF resume.")
            return
        if not jd_input.strip():
            st.warning("Please paste the job description text.")
            return
            
        with st.spinner("Processing documents and computing similarity vectors..."):
            try:
                encoder = load_model()
                nlp = load_spacy()
                
                jd_vec = get_embedding(jd_input, encoder)
                final_kws = extract_keywords(jd_input, nlp)
                
                results = []
                for f in files:
                    res = process_file(f, jd_input, jd_vec, final_kws, encoder, nlp)
                    results.append(res)
                    
                # sort highest scores first
                valid_res = [r for r in results if not r.get("error") or r.get("is_warning")]
                valid_res.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                
            except Exception as e:
                st.error(f"Could not analyze documents properly. An unexpected error occurred: {str(e)}")
                return
            
        st.success("Analysis Complete")
        st.divider()
        
        # display any critical errors that completely broke parsing
        for r in results:
            if r.get("error") and not r.get("is_warning"):
                st.error(f"{r['filename']}: {r['error']}")
                
        if not valid_res:
            st.info("No valid resumes could be fully processed.")
            return
            
        st.header("Ranked Results")
        
        for idx, item in enumerate(valid_res):
            is_best_match = (idx == 0)
            
            # soft warnings (e.g. looks like a syllabus, not a resume, but parsed anyway)
            if item.get("is_warning"):
                st.warning(f"{item['filename']}: {item['error']}")
                
            scores = item.get("scores_dict", {})
            matched = item.get("matched_keywords", [])
            missing = item.get("missing_keywords", [])
            total_score = item.get("score", 0.0)
            missing_secs = item.get("missing_sections", [])
            
            title = f"{item['filename']} - Match Score: {total_score:.1f}%"
            if is_best_match and len(valid_res) >= 1:
                st.markdown("### TOP MATCH")
                
            with st.expander(title, expanded=is_best_match):
                
                if total_score >= 80:
                    color = "green"
                elif total_score >= 50:
                    color = "orange"
                else:
                    color = "red"
                    
                st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 3em;'>Match Score: {total_score:.1f}%</h1>", unsafe_allow_html=True)
                
                st.divider()
                st.markdown("#### Section Scores")
                c_scores = st.columns(3)
                c_scores[0].metric("Skills", f"{scores.get('skills', 0):.1f}%")
                c_scores[1].metric("Experience", f"{scores.get('experience', 0):.1f}%")
                
                if "projects" in missing_secs:
                    st.error("Projects Score: 0.0% (No projects section found)")
                else:
                    c_scores[2].metric("Projects", f"{scores.get('projects', 0):.1f}%")
                    
                st.divider()
                
                total_kws = len(matched) + len(missing)
                st.markdown("#### Matched vs Missing Skills")
                kw_ratio = int((len(matched) / total_kws * 100)) if total_kws > 0 else 100
                st.progress(kw_ratio, text=f"Matched {len(matched)} / {total_kws} skills")
                
                c_match, c_miss = st.columns(2)
                with c_match:
                    if matched:
                        for k in matched:
                            st.markdown(f"- :green[{k.title()}]")
                    else:
                        st.markdown("- None identified")
                        
                with c_miss:
                    if missing:
                        for k in missing:
                            st.markdown(f"- :red[{k.title()}]")
                    else:
                        st.markdown("- None. All required skills met!")
                st.divider()
                
                # temporary pipeline debug view for interview walkthroughs
                with st.expander("Debug Output"):
                    exact = list(set(matched) - set(item.get("partial_keywords", [])) - set(item.get("semantic_keywords", [])))
                    st.write("**Target JD Keywords:**", final_kws)
                    st.write("**Engine Match Breakdown:**")
                    st.write(f"- Exact String Substring Matches: {exact}")
                    st.write(f"- Partial Root Stem Matches: {item.get('partial_keywords', [])}")
                    st.write(f"- Semantic Vector Fallback Matches (>60%): {item.get('semantic_keywords', [])}")
                    st.write(f"- Failed to Find: {missing}")
                
                guides = item.get("score_guidance", [])
                if guides:
                    st.markdown("#### How to Improve Your Score")
                    for g in guides:
                        st.markdown(f"- {g}")
                    st.divider()
                
                # only query gemini for the top rank to save compute
                fb_data = {
                    "strengths": item.get("strengths", []),
                    "improvements": item.get("improvements", [])
                }
                
                if is_best_match:
                    fb_data = analyze_with_gemini(scores, missing, item.get("resume_text", ""), jd_input, fb_data)
                    
                st.markdown("#### Strengths")
                for s in fb_data.get('strengths', []):
                    st.markdown(f"- {s}")
                    
                st.markdown("#### Improvement Suggestions")
                for i in fb_data.get('improvements', []):
                    st.markdown(f"- {i}")
                    
                with st.expander("Apply for Similar Jobs"):
                    st.write("[Internshala](https://internshala.com)")
                    st.write("[LinkedIn Jobs](https://www.linkedin.com/jobs/)")
                    st.write("[Glassdoor](https://www.glassdoor.co.in/Job/index.htm)")
                    st.write("[Indeed](https://in.indeed.com)")

if __name__ == "__main__":
    main()
