import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# Page setup
st.set_page_config(page_title="📄 QP & Rubric Upload Portal", layout="centered")
st.title("📚 Question Paper & Rubric Upload System")

# Directories
qp_dir = "uploaded/question_papers"
rubric_dir = "uploaded/rubrics"
Path(qp_dir).mkdir(parents=True, exist_ok=True)
Path(rubric_dir).mkdir(parents=True, exist_ok=True)

st.markdown("### 📥 Upload Section")

# File Uploaders
qp_file = st.file_uploader("Upload Question Paper (PDF/DOCX)", type=["pdf", "docx"], key="qp")
rubric_file = st.file_uploader("Upload Rubric (Excel Only)", type=["xlsx"], key="rubric")

# Function to save file if not already uploaded
def save_unique_file(uploaded_file, folder):
    filename = uploaded_file.name
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        st.warning(f"⚠️ File '{filename}' already exists. Upload skipped.")
        return None
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

# Handle Question Paper upload
if qp_file:
    qp_path = save_unique_file(qp_file, qp_dir)
    if qp_path:
        st.success(f"✅ Question Paper uploaded: {qp_file.name}")

# Handle Rubric upload
if rubric_file:
    rubric_path = save_unique_file(rubric_file, rubric_dir)
    if rubric_path:
        st.success(f"✅ Rubric uploaded: {rubric_file.name}")

# Display Question Papers
st.markdown("### 📰 Uploaded Question Papers")
qp_files = sorted(os.listdir(qp_dir))
if qp_files:
    for file in qp_files:
        file_path = os.path.join(qp_dir, file)
        with open(file_path, "rb") as f:
            st.download_button(label=f"⬇️ {file}", data=f, file_name=file)
else:
    st.info("No question papers uploaded yet.")

# Display Rubrics
st.markdown("### 📊 Uploaded Rubrics")
rubric_files = sorted(os.listdir(rubric_dir))
if rubric_files:
    for file in rubric_files:
        file_path = os.path.join(rubric_dir, file)
        with open(file_path, "rb") as f:
            st.download_button(label=f"⬇️ {file}", data=f, file_name=file)
else:
    st.info("No rubrics uploaded yet.")