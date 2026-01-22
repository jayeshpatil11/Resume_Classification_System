
import streamlit as st
import docx2txt
import fitz
import mammoth
import tempfile
import pickle

@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

def read_pdf(path):
    text = ""
    try:
        pdf = fitz.open(path)
        for page in pdf:
            text += page.get_text()
    except:
        pass
    return text

def read_docx(path):
    try:
        return docx2txt.process(path)
    except:
        return ""

def read_doc(path):
    try:
        with open(path, "rb") as file:
            result = mammoth.extract_raw_text(file)
        return result.value
    except:
        return ""

def extract_text(file):
    ext = file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file.read())
        temp_path = temp.name

    if ext == "pdf":
        return read_pdf(temp_path)
    elif ext == "docx":
        return read_docx(temp_path)
    elif ext == "doc":
        return read_doc(temp_path)
    else:
        return ""

st.title("📄 Resume Classification (Folder-Based Categories)")
st.write("Predictions are based on your dataset’s folder names.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "doc", "docx"])

if uploaded_file:
    extracted = extract_text(uploaded_file)

    if not extracted.strip():
        st.error("Could not extract text.")
    else:
        features = vectorizer.transform([extracted])
        prediction = model.predict(features)[0]

        st.subheader("📌 Predicted Category:")
        st.markdown(f"## 🎯 {prediction}")


#COMMAND
#python preprocess.py
#python train_model.py
#streamlit run app.py

