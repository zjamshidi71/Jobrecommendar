import re
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="JobMatcher Pro", layout="centered")


# Your Adzuna keys. Keep the quotes around each value.
APP_ID = "c439c644"
APP_KEY = "01163d7f667bc6da157e36a54a5e2abd"


def preprocess_text(text):
    # Lowercase and strip punctuation; stop words are removed by the vectorizer.
    return re.sub(r"[^a-z0-9+#\s]", " ", (text or "").lower())


# Cached so moving a slider doesn't call the API again (saves your quota).
@st.cache_data(ttl=3600, show_spinner="Searching Adzuna...")
def fetch_jobs(keywords, location="Calgary", min_salary=None, full_time=True):
    url = "https://api.adzuna.com/v1/api/jobs/ca/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 50,
        "what": keywords,
    }
    if location:
        params["where"] = location
    if min_salary:
        params["salary_min"] = min_salary
    if full_time:
        params["full_time"] = 1
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"Adzuna returned {response.status_code}: {response.text[:200]}")
    return response.json().get("results", [])


@st.cache_data
def read_resume(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_recommendations(jobs, resume_text, ratings, top_n=10, feedback_strength=0.5):
    """Rank jobs by similarity to the resume, adjusted by the user's ratings.

    A rating above 3 pulls up jobs similar to the rated job; below 3 pushes them down.
    """
    docs = [preprocess_text(f"{j['title']} {j['description']}") for j in jobs]
    query = preprocess_text(resume_text)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs + [query])
    job_vecs, query_vec = X[:-1], X[-1]

    scores = cosine_similarity(job_vecs, query_vec).ravel()

    if ratings:
        job_sim = cosine_similarity(job_vecs)  # job-to-job similarity
        for idx, rating in ratings.items():
            weight = (rating - 3) / 2  # 1 -> -1, 3 -> 0, 5 -> +1
            scores = scores + feedback_strength * weight * job_sim[:, idx]

    order = np.argsort(-scores)
    rated = set(ratings)
    return [(jobs[i], scores[i]) for i in order if i not in rated][:top_n]


def show_job(job):
    st.markdown(f"**{job['title']}**")
    company = job.get("company", {}).get("display_name", "")
    location = job.get("location", {}).get("display_name", "")
    st.caption(" · ".join(x for x in [company, location] if x))
    st.write(job["description"][:250] + "...")
    st.markdown(f"[Apply here]({job['redirect_url']})")


# ---- Layout ----
st.title("🔍 JobMatcher Pro")
st.write("Find jobs that match your skills and interests!")

st.subheader("Enter your job search criteria")
keywords = st.text_input("Keywords or fields of interest", placeholder="e.g., Data Science, Machine Learning, AI")
col1, col2 = st.columns(2)
location = col1.text_input("Location", value="Calgary")
full_time = col2.checkbox("Full-time only", value=True)

st.subheader("Upload your resume")
uploaded_file = st.file_uploader("Upload a PDF of your resume", type="pdf")

if not (uploaded_file and keywords):
    st.info("Please enter your search criteria and upload your resume.")
    st.stop()

resume_text = read_resume(uploaded_file.getvalue())
if not resume_text.strip():
    st.error("No text could be read from that PDF. It may be a scanned image.")
    st.stop()

try:
    jobs = fetch_jobs(keywords, location, full_time=full_time)
except Exception as e:
    st.error(f"Job search failed: {e}")
    st.stop()

if not jobs:
    st.error("No jobs found with the specified criteria.")
    st.stop()

st.success(f"Found {len(jobs)} job listings matching '{keywords}'.")

st.subheader("Rate a few jobs")
st.caption("3 is neutral. Higher ratings pull similar jobs up; lower ratings push them down.")
ratings = {}
for i, job in enumerate(jobs[:3]):
    with st.container(border=True):
        show_job(job)
        # A unique key per job avoids Streamlit's duplicate-widget error.
        ratings[i] = st.slider("Your rating", 1, 5, 3, key=f"rate_{job['id']}")

st.subheader("Top job matches for your profile")
for job, score in get_recommendations(jobs, resume_text, ratings):
    st.divider()
    show_job(job)
