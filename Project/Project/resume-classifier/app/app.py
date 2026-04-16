import streamlit as st
import sys
import os
import tempfile

# add project root to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.predict import load_model_and_vectorizer, predict_role
from src.preprocess import clean_text
from utils.pdf_parser import extract_text_from_pdf

# page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="_",
    layout="centered"
)

st.title("Resume Classifier")
st.write("Upload a resume (PDF or paste text) and the system will predict the most suitable job role.")

# load model once and cache it so it doesn't reload every time
@st.cache_resource
def get_model():
    return load_model_and_vectorizer()

try:
    model, vectorizer = get_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model. Make sure you've trained it first by running train_model.py\nError: {e}")

st.markdown("---")

# two options - upload pdf or paste text
tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF resume", type=['pdf'])

    if uploaded_file is not None and model_loaded:
        # save uploaded file temporarily so pdfplumber can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # extract text from pdf
        resume_text = extract_text_from_pdf(tmp_path)

        # clean up temp file
        os.unlink(tmp_path)

        if resume_text.strip():
            st.subheader("Extracted Text Preview")
            with st.expander("Click to see extracted text"):
                st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))

            if st.button("Classify Resume", key="pdf_btn"):
                with st.spinner("Analyzing resume..."):
                    prediction, confidence, top_preds = predict_role(resume_text, model, vectorizer)

                st.success(f"Predicted Role: **{prediction}**")
                if confidence is not None:
                    st.metric("Confidence", f"{confidence:.2%}")

                # show top predictions
                st.subheader("All Predictions")
                for role, score in top_preds:
                    if score != "N/A":
                        st.write(f"- {role}: {score:.4f}")
                    else:
                        st.write(f"- {role}")

                # show some keywords from the resume
                st.subheader("Key Terms Found")
                cleaned = clean_text(resume_text)
                words = cleaned.split()
                # get the most frequent meaningful words
                from collections import Counter
                word_counts = Counter(words).most_common(15)
                keywords = [w for w, c in word_counts]
                st.write(", ".join(keywords))
        else:
            st.warning("Could not extract text from this PDF. It might be a scanned document.")

with tab2:
    text_input = st.text_area(
        "Paste resume text here",
        height=250,
        placeholder="Paste the resume content here..."
    )

    if st.button("Classify", key="text_btn") and text_input.strip() and model_loaded:
        with st.spinner("Analyzing..."):
            prediction, confidence, top_preds = predict_role(text_input, model, vectorizer)

        st.success(f"Predicted Role: **{prediction}**")
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.2%}")

        st.subheader("All Predictions")
        for role, score in top_preds:
            if score != "N/A":
                st.write(f"- {role}: {score:.4f}")
            else:
                st.write(f"- {role}")

        # keywords
        st.subheader("Key Terms Found")
        cleaned = clean_text(text_input)
        words = cleaned.split()
        from collections import Counter
        word_counts = Counter(words).most_common(15)
        keywords = [w for w, c in word_counts]
        st.write(", ".join(keywords))

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Resume Classifier - NLP Project | B.Tech CSE"
    "</div>",
    unsafe_allow_html=True
)
