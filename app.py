import streamlit as st
from textblob import TextBlob

# ----- Page config -----
st.set_page_config(
    page_title="Text Sentiment Analyzer",
    layout="centered"
)

# ----- Theme handling -----
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

st.title("📝 Text Sentiment Analyzer")

theme = st.radio(
    "Choose theme:",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1,
    horizontal=True
)
st.session_state.theme = theme

# Colors for themes
if theme == "Light":
    bg_color = "#e0e0e0"        # grey background
    text_color = "#111111"
    card_color = "#f7f7f7"
    border_color = "#cccccc"
else:
    bg_color = "#202124"        # dark grey background
    text_color = "#f5f5f5"
    card_color = "#303134"
    border_color = "#5f6368"

# ----- Custom CSS -----
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        color: {text_color};
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    .sent-card {{
        background-color: {card_color};
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid {border_color};
        margin-top: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----- Input -----
user_text = st.text_area(
    "Enter some text to analyze:",
    height=150,
    placeholder="Type your sentence or paragraph here..."
)

analyze = st.button("Analyze Sentiment")

# ----- Analysis -----
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity   # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Label for polarity
        if polarity > 0.1:
            label = "Positive 😊"
        elif polarity < -0.1:
            label = "Negative 😞"
        else:
            label = "Neutral 😐"

        st.markdown(
            f"""
            <div class="sent-card">
                <h3 style="margin-bottom: 0.5rem;">Sentiment Result</h3>
                <p><b>Label:</b> {label}</p>
                <p><b>Polarity:</b> {polarity:.3f}</p>
                <p><b>Subjectivity:</b> {subjectivity:.3f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
