Live Demo
Try the app here: [Chat with your PDFs]
(https://rag-pdf-chat-ipsqxi2nnqqtbezedtxk6b.streamlit.app/)

# Chat with your PDFs

A free AI app that lets you upload one or more PDFs, chat with them, and turn them into study notes. Every answer shows the exact file and page it came from, so you can trust it.

## The problem it solves

Students and professionals lose hours searching long PDFs for one answer. This app finds the answer in seconds and shows its proof, and says so honestly when the answer is not in the documents.

## Features

- Upload multiple PDFs and search across all of them at once
- Continuous chat with memory, so follow-up questions work
- Source proof: each answer shows the file name, page number and the exact text used
- Honest answers: if the information is not in the documents, it says so instead of guessing
- Study tools: summary, key points, quiz questions and flashcards
- Works with questions typed in simple or informal English

## How it works

This app uses Retrieval-Augmented Generation (RAG). The PDFs are split into small chunks, turned into embeddings, and stored in a FAISS search index. When you ask a question, the closest chunks are found and sent to a Groq language model, which answers using only that text.

## Tech stack

Python, Streamlit, pypdf, sentence-transformers (all-MiniLM-L6-v2), FAISS, Groq API

## Run it on your computer

Clone the repository and open the folder. Create and activate a virtual environment. Install the packages with pip: streamlit, pypdf, faiss-cpu, python-dotenv, groq, sentence-transformers and torchvision. Create a file named .env containing GROQ_API_KEY=your_key_here. A free key is available from the Groq console. Then run: streamlit run app.py

## Screenshots

Add a screenshot of the Chat tab and the Study tools tab here.

## What I plan to add next

- Evaluation set to measure answer accuracy
- Live public demo

## Author

Neethu Jimmy Joy, BTech CSE (AIML), SRM Institute of Science and Technology, Vadapalani