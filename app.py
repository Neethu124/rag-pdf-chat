import os
import numpy as np
import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. Load the Groq key from the .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHAT_MODEL = "openai/gpt-oss-20b"


# 2. Load a small free model that runs on your laptop to make embeddings
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 3. Read the PDF and split each page into small chunks
def read_pdf(file):
    reader = PdfReader(file)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 800):
            piece = text[i:i + 800].strip()
            if piece:
                chunks.append({"text": piece, "page": page_number})
    return chunks


# 4. Turn text into numbers (embeddings) so we can compare meaning
def embed(texts):
    model = load_embedder()
    vectors = model.encode(texts)
    return np.array(vectors, dtype="float32")


# 5. Build a FAISS index so we can search quickly
@st.cache_resource(show_spinner=False)
def build_index(file_name, _chunks):
    vectors = embed([c["text"] for c in _chunks])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


# 6. Find the chunks closest in meaning to the question
def search(question, index, chunks, k=4):
    q_vector = embed([question])
    _, ids = index.search(q_vector, k)
    return [chunks[i] for i in ids[0]]


# 7. Ask Groq to answer using only those chunks
def answer(question, found_chunks):
    context = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in found_chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you could not find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# 8. The Streamlit web page
st.title("Chat with your PDF")
st.write("Upload a PDF, ask a question, and get an answer with page numbers.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    chunks = read_pdf(uploaded)
    with st.spinner("Reading your document..."):
        index = build_index(uploaded.name, chunks)
    st.success(f"Ready! Split into {len(chunks)} chunks.")

    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Thinking..."):
            found = search(question, index, chunks)
            reply = answer(question, found)
        st.subheader("Answer")
        st.write(reply)
        pages = sorted({c["page"] for c in found})
        st.caption("Sources: pages " + ", ".join(map(str, pages)))