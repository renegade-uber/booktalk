# BookTalk Prototype (Local Console Version with PDF Support + Semantic Search + GPT Synthesis)

# Goal: Ingest a large e-book (PDF or TXT) and allow users to ask questions about it
# using semantic search via vector embeddings and FAISS, then synthesize an answer using GPT.

import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Step 1: Load book text (supports TXT and PDF)
def load_book(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type: must be .txt or .pdf")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Step 2: Split text into chunks (e.g. ~300 words)
def chunk_text(text, max_words=300):
    paragraphs = text.split("\n")
    chunks, current_chunk = [], []
    word_count = 0
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.extend(words)
        word_count += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Step 3: Build FAISS index
def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

# Step 4: Answer questions using semantic similarity
def answer_question_faiss(question, chunks, model, index, top_k=3):
    query_vec = model.encode([question])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

# Step 5: Generate answer using OpenAI GPT (openai>=1.0.0 compatible)
def gpt_answer(question, context_chunks):
    from openai import OpenAI
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    prompt = f"""You are a helpful assistant. ONLY use the information provided in the context below to answer the user's question. 
Do not add any outside knowledge or make assumptions. If the answer is not in the context, respond with "I don't know based on the provided information."

Context:
{chr(10).join(context_chunks)}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# Step 6: Command-line interface
def main():
    book_path = input("Enter path to the .txt or .pdf file of your book: ").strip()
    if not os.path.exists(book_path):
        print("File not found.")
        return

    try:
        book_text = load_book(book_path)
    except Exception as e:
        print(f"Error loading book: {e}")
        return

    print("Splitting text and building vector index...")
    chunks = chunk_text(book_text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, _ = build_faiss_index(chunks, model)

    print("Book loaded and indexed successfully. Ask your question below (type 'exit' to quit).\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ('exit', 'quit'): break
        answers = answer_question_faiss(question, chunks, model, index)
        print("\n--- Source Chunks ---")
        for a in answers:
            print(f"\n{a}\n")
        print("\n--- GPT Answer ---")
        try:
            final_answer = gpt_answer(question, answers)
            print(f"\n{final_answer}\n")
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")

if __name__ == '__main__':
    main()
