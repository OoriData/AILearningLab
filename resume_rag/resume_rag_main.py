import asyncio
import os
import argparse
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import httpx
from quart import Quart, request, jsonify, render_template
import aiofiles

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the RAG application')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to show complete prompts')
args = parser.parse_args()

# Initialize Quart app
app = Quart(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_MODE'] = args.debug
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB variables
client = None
collection = None

# PDF Processing Functions
async def extract_text_from_pdf(pdf_path):
    """Asynchronously extract text from a PDF file."""
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, lambda: _extract_text_sync(pdf_path))
    return text, pdf_path

def _extract_text_sync(pdf_path):
    """Synchronously extract text from a PDF file."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

async def process_pdf_directory(directory_path):
    """Process all PDFs in a directory asynchronously."""
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                if f.lower().endswith('.pdf')]
    
    tasks = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_files]
    results = await asyncio.gather(*tasks)
    
    document_texts = {os.path.basename(path): text for text, path in results}
    return document_texts

# Text Chunking Functions
async def split_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks asynchronously."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        if end < text_length and text[end] != ' ':
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

async def process_documents(doc_texts):
    """Process documents into chunks asynchronously."""
    all_chunks = []
    
    for filename, text in doc_texts.items():
        chunks = await split_text(text)
        doc_chunks = [(filename, i, chunk) for i, chunk in enumerate(chunks)]
        all_chunks.extend(doc_chunks)
    
    return all_chunks

# ChromaDB Functions
async def setup_chroma():
    """Set up ChromaDB asynchronously."""
    loop = asyncio.get_event_loop()
    client = await loop.run_in_executor(
        None, 
        lambda: chromadb.Client(Settings(persist_directory="./chroma_db"))
    )
    
    collection = await loop.run_in_executor(
        None,
        lambda: client.get_or_create_collection("document_chunks")
    )
    
    return client, collection

async def add_chunks_to_chroma(collection, chunks):
    """Add document chunks to ChromaDB asynchronously."""
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": idx} for filename, idx, _ in chunks]
    documents = [chunk for _, _, chunk in chunks]
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )
    )

# Ollama Functions
async def query_ollama(prompt, model="llama2"):
    """Query Ollama API asynchronously."""
    if app.config['DEBUG_MODE']:
        print("\n===== OLLAMA PROMPT =====")
        print(prompt)
        print("========================\n")
        
    async with httpx.AsyncClient() as client:
        response = await client.post(
            # "http://localhost:11434/api/generate",
            "http://localhost:1234/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "")

# LM Studio Functions
async def query_lm_studio(prompt, model=None):
    """Query LM Studio API asynchronously."""
    if app.config['DEBUG_MODE']:
        print("\n===== LM STUDIO PROMPT =====")
        print(prompt)
        print("============================\n")
        
    async with httpx.AsyncClient() as client:
        # Set the base URL for LM Studio
        base_url = "http://localhost:1234/v1"
        
        # If no model is specified, LM Studio will use the default loaded model
        if model:
            # Specify the model if needed
            messages = [
                {"role": "system", "content": f"Use model: {model}"},
                {"role": "user", "content": prompt},
            ]
        else:
            # Use the default model
            messages = [
                {"role": "user", "content": prompt},
            ]
        
        # Debug the actual messages being sent
        if app.config['DEBUG_MODE']:
            print("Messages being sent to LM Studio:")
            for msg in messages:
                print(f"Role: {msg['role']}, Content: {msg['content'][:100]}...")
        
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model if model else None,  # LM Studio will use the default model if not specified
                "messages": messages,
            }
        )
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

# RAG Query Processing
async def process_rag_query(query, collection, model="llama2"):
    """Process a RAG query asynchronously."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: collection.query(
            query_texts=[query],
            n_results=3
        )
    )
    
    retrieved_chunks = results["documents"][0]
    chunk_sources = [meta["source"] for meta in results["metadatas"][0]]
    
    prompt = f"""
    Answer the following question based on the provided context.
    If you don't know the answer from the context, say "I don't have enough information to answer that."
    
    Context:
    {' '.join(retrieved_chunks)}
    
    Question: {query}
    
    Answer:
    """

    # query_handler = query_ollama
    query_handler = query_lm_studio

    response = await query_handler(prompt, model)
    
    return {
        "response": response,
        "sources": chunk_sources,
        "prompt": prompt if app.config['DEBUG_MODE'] else None
    }

# Quart Routes
@app.before_serving
async def setup():
    """Set up ChromaDB before serving requests."""
    global client, collection
    client, collection = await setup_chroma()

@app.route('/')
async def index():
    """Render the home page."""
    return await render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload_file():
    """Handle file upload."""
    files = await request.files
    
    if 'file' not in files:
        return jsonify({"error": "No file part"}), 400
    
    uploaded_files = files.getlist('file')
    
    for file in uploaded_files:
        if file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            await file.save(file_path)
    
    doc_texts = await process_pdf_directory(app.config['UPLOAD_FOLDER'])
    chunks = await process_documents(doc_texts)
    await add_chunks_to_chroma(collection, chunks)
    
    return jsonify({"message": "Files uploaded and processed successfully"})

@app.route('/chat', methods=['POST'])
async def chat():
    """Handle chat requests."""
    data = await request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    result = await process_rag_query(query, collection)
    return jsonify(result)

if __name__ == '__main__':
    if app.config['DEBUG_MODE']:
        print("Debug mode enabled - prompts will be displayed")
    app.run(debug=True)