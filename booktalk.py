# BookTalk Prototype (Local Console Version with PDF Support + Semantic Search + GPT Synthesis)

# Goal: Ingest a large e-book (PDF or TXT) and allow users to ask questions about it
# using semantic search via vector embeddings and FAISS, then synthesize an answer using GPT.

import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
def answer_question_faiss(question, chunks, model, index, top_k=10):
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
When answering questions, generate thorough, multi-paragraph responses that cite and summarize all relevant context chunks. Do not be brief.
Start with a short summary answering the question, then provide additional details and context from the provided information.
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

#
# Application layer
# 

# Terminal application
def main():
    books_folder = os.path.expanduser("~/Documents/books")
    book_files = [f for f in os.listdir(books_folder) if f.endswith(('.pdf', '.txt'))][:9]
    
    if not book_files:
        print("No PDF or TXT files found in your books folder.")
        return
    
    print("Available books:")
    for i, book in enumerate(book_files, 1):
        print(f"{i}. {book}")
    
    selection = input("Enter book numbers to index (1,2,3) or leave empty for all: ").strip()
    
    if selection:
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_books = [book_files[idx] for idx in selected_indices if 0 <= idx < len(book_files)]
        except (ValueError, IndexError):
            print("Invalid selection. Using all available books.")
            selected_books = book_files
    else:
        selected_books = book_files
    
    all_chunks = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for book_file in selected_books:
        book_path = os.path.join(books_folder, book_file)
        print(f"Loading {book_file}...")
        try:
            book_text = load_book(book_path)
            chunks = chunk_text(book_text)
            all_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from {book_file}")
        except Exception as e:
            print(f"Error loading {book_file}: {e}")
    
    if not all_chunks:
        print("No books were successfully loaded.")
        return
    
    print("Building vector index...")
    index, _ = build_faiss_index(all_chunks, model)
    print("Books indexed successfully. Ask your question below (type 'exit' to quit).\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ('exit', 'quit'): break
        answers = answer_question_faiss(question, all_chunks, model, index)
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

# Chainlit entry point
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    books_folder = os.path.expanduser("~/Documents/books")
    book_files = [f for f in os.listdir(books_folder) if f.endswith(('.pdf', '.txt'))][:9]
    
    if not book_files:
        await cl.Message(content="No PDF or TXT files found in your books folder.").send()
        return
    
    # Show book list
    book_list = "\n".join([f"- {book}" for book in book_files])
    await cl.Message(content=f"Welcome to BookTalk! Indexing all available books:\n\n{book_list}").send()
    
    # Show loading message
    loading_msg = cl.Message(content=f"Indexing {len(book_files)} books...")
    await loading_msg.send()
    
    # Index books
    all_chunks = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for book_file in book_files:
        book_path = os.path.join(books_folder, book_file)
        try:
            book_text = load_book(book_path)
            chunks = chunk_text(book_text)
            all_chunks.extend(chunks)
            await loading_msg.stream_token(f"\nAdded {len(chunks)} chunks from {book_file}")
        except Exception as e:
            await loading_msg.stream_token(f"\nError loading {book_file}: {str(e)}")
    
    if not all_chunks:
        await cl.Message(content="No books were successfully loaded.").send()
        return
    
    # Build index
    await loading_msg.stream_token("\nBuilding vector index...")
    index, _ = build_faiss_index(all_chunks, model)
    
    await loading_msg.stream_token("\nBooks indexed successfully. Ask your questions!")
    cl.user_session.set("all_chunks", all_chunks)
    cl.user_session.set("index", index)
    cl.user_session.set("model", model)

@cl.on_message
async def on_message(message: cl.Message):
    # Get session variables
    all_chunks = cl.user_session.get("all_chunks")
    index = cl.user_session.get("index")
    model = cl.user_session.get("model")
    
    if not all_chunks or index is None:
        await cl.Message(content="Please wait for book indexing to complete.").send()
        return
    
    question = message.content
    
    # Get relevant chunks
    answers = answer_question_faiss(question, all_chunks, model, index)
    
    # Show source chunks in expandable element
    sources_content = "\n\n".join([f"Source {i+1}:\n{chunk}" for i, chunk in enumerate(answers)])
    sources = cl.Text(name="Sources", content=sources_content, display="inline")
    
    # Generate GPT answer
    try:
        # Create a new message instead of updating
        await cl.Message(content="Generating answer...").send()
        
        final_answer = gpt_answer(question, answers)
        # Send a new message with the answer
        await cl.Message(content=final_answer, elements=[sources]).send()
    except Exception as e:
        await cl.Message(content=f"Error calling OpenAI API: {str(e)}").send()
