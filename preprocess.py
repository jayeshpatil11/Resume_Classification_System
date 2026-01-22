import os
import pandas as pd
import docx2txt
import fitz
import mammoth

DATASET_PATH = "Resume"   # <--- your main folder containing subfolders

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

def extract_text(filepath):
    ext = filepath.split(".")[-1].lower()
    if ext == "pdf":
        return read_pdf(filepath)
    elif ext == "docx":
        return read_docx(filepath)
    elif ext == "doc":
        return read_doc(filepath)
    else:
        return ""

data = []

for category in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        text = extract_text(file_path)

        if text.strip():
            data.append([filename, category, text])

df = pd.DataFrame(data, columns=["filename", "category", "content"])
df.to_csv("processed_resume_dataset.csv", index=False)

print("Dataset created successfully!")
