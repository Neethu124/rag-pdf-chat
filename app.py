import os
import numpy as np
import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. Load the Groq key. Locally it comes from .env.
#    When hosted online, it comes from Streamlit secrets.
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

client = Groq(api_key=api_key)
CHAT_MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")


# 2. Small free model on the server that turns text into embeddings
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 3. Read one PDF and split each page into chunks with file name and page
def read_pdf(file):
    reader = PdfReader(file)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 800):
            piece = text[i:i + 800].strip()
            if piece:
                chunks.append(
                    {"text": piece, "page": page_number, "file": file.name}
                )
    return chunks


def embed(texts):
    model = load_embedder()
    vectors = model.encode(texts)
    return np.array(vectors, dtype="float32")


# 4. Build one search index across all uploaded PDFs
@st.cache_resource(show_spinner=False)
def build_index(file_key, _chunks):
    vectors = embed([c["text"] for c in _chunks])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def search(question, index, chunks, k=4):
    q_vector = embed([question])
    _, ids = index.search(q_vector, k)
    return [chunks[i] for i in ids[0]]


# 5. One helper to talk to Groq
def ask_llm(messages):
    response = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages
    )
    return response.choices[0].message.content


# 6. Chat answer: uses found chunks plus the last few messages as memory
def answer(question, found_chunks, history):
    context = "\n\n".join(
        f"[{c['file']}, page {c['page']}] {c['text']}" for c in found_chunks
    )
    system = (
        "You answer questions about the user's PDFs. Use ONLY the context "
        "given. Mention the file name and page number for facts you use. "
        "If the answer is not in the context, reply exactly: "
        "'I could not find this in the uploaded documents.' "
        "Do not guess."
    )
    messages = [{"role": "system", "content": system}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )
    return ask_llm(messages)


# 7. Study tools work on the start of each document
def document_text(chunks, limit=12000):
    text = ""
    for c in chunks:
        text += f"[{c['file']}, page {c['page']}] {c['text']}\n"
        if len(text) > limit:
            break
    return text[:limit]


def study_tool(task, chunks):
    prompts = {
        "summary": "Write a clear summary of this document in about 8 short bullet points.",
        "keypoints": "List the 10 most important key points a student should remember.",
        "quiz": "Write 5 multiple choice questions with 4 options each. Put the correct answers at the end.",
        "flashcards": "Make 8 flashcards. Format each as: Q: question, then A: answer.",
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful study assistant. Use ONLY the document text given.",
        },
        {
            "role": "user",
            "content": prompts[task] + "\n\nDocument:\n" + document_text(chunks),
        },
    ]
    return ask_llm(messages)


# 8. The page
st.title("Chat with your PDFs")
st.write("Upload one or more PDFs, chat with them, or turn them into study notes.")

uploaded_files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files:
    all_chunks = []
    for f in uploaded_files:
        all_chunks.extend(read_pdf(f))

    if not all_chunks:
        st.warning("No readable text found. Scanned PDFs are not supported yet.")
        st.stop()

    file_key = "|".join(sorted(f.name for f in uploaded_files))
    with st.spinner("Reading your documents..."):
        index = build_index(file_key, all_chunks)
    st.success(f"Ready! {len(uploaded_files)} file(s), {len(all_chunks)} chunks.")

    # Clear the chat when a different set of files is uploaded
    if st.session_state.get("file_key") != file_key:
        st.session_state.file_key = file_key
        st.session_state.messages = []

    tab_chat, tab_study = st.tabs(["Chat", "Study tools"])

    with tab_chat:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("sources"):
                    with st.expander("Sources"):
                        for s in m["sources"]:
                            st.markdown(f"**{s['file']} - page {s['page']}**")
                            st.write(s["text"])

        question = st.chat_input("Ask a question about your PDFs")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    found = search(question, index, all_chunks)
                    reply = answer(question, found, st.session_state.messages)
                st.write(reply)
                with st.expander("Sources"):
                    for s in found:
                        st.markdown(f"**{s['file']} - page {s['page']}**")
                        st.write(s["text"])
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "sources": found}
            )

    with tab_study:
        st.write("Pick a tool. It reads the first part of your documents.")
        col1, col2 = st.columns(2)
        choice = None
        if col1.button("Summarize"):
            choice = "summary"
        if col2.button("Key points"):
            choice = "keypoints"
        if col1.button("Quiz me"):
            choice = "quiz"
        if col2.button("Flashcards"):
            choice = "flashcards"
        if choice:
            with st.spinner("Working on it..."):
                st.markdown(study_tool(choice, all_chunks))
else:
    st.info("Upload at least one PDF to begin.")