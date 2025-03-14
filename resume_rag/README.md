# TL;DR Version: Building a Simple Resume RAG Tool with Python, Ollama, ChromaDB, and Quartz

Make sure you have Python 3.12+ & Ollama installed, as in the main README, then set up a virtual environment:

## Creating a Virtual Environment

First, open a terminal or command prompt and navigate to your project directory. Run the following commands to create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv resume_rag_venv

# Activate the virtual environment (macOS/Linux)
source resume_rag_venv/bin/activate

# Activate the virtual environment (Windows)
# resume_rag_venv\Scripts\activate
```

## Run the code

```sh
python resume_rag_main.py
```

If you're not getting great responses from your LLM you can get some debug traces as follows, though it may take a bit of a practiced eye to apply what it shows:

```sh
python resume_rag_main.py --debug 
```

---

# Full version: Building a Simple Resume RAG Tool with Python, Ollama, ChromaDB, and Quartz

A comprehensive guide to help AI development novices build a Retrieval-Augmented Generation (RAG) system on their laptops using Python, Ollama, ChromaDB, and Quartz for the web interface. RAG systems enhance large language model outputs by retrieving relevant information from a knowledge base before generating responses. This implementation prioritizes asynchronous functions to teach good programming practices from the start while maintaining simplicity for beginners. By the end of this guide, you'll have a functional RAG system that allows users to chat with their documents through an intuitive web interface.

## Understanding RAG Architecture

Retrieval-Augmented Generation represents a powerful approach to enhancing AI systems by combining the strengths of retrieval-based and generation-based methods. RAG systems first retrieve relevant information from a knowledge base (in our case, user-provided PDF documents) and then use that information to generate contextually appropriate responses. This architecture significantly improves the accuracy and relevance of AI-generated content by grounding responses in specific documents rather than relying solely on the model's pre-trained knowledge. The core components of our RAG implementation include document processing for text extraction, vector storage in ChromaDB, similarity search to find relevant context, and response generation using Ollama's language models.

RAG systems are particularly valuable when working with domain-specific information or when answers need to be traceable to source documents. Unlike traditional chatbots that can hallucinate or provide outdated information, RAG systems can maintain accuracy by referencing specific documents and even citing sources when needed. Our implementation will create a complete pipeline from document ingestion to user interaction, all using asynchronous programming patterns to ensure responsive performance even with larger document collections.

### Project Components Overview

Each component in our RAG system serves a specific purpose in the information retrieval and generation pipeline. Python provides the programming foundation, with its asyncio library enabling efficient handling of I/O-bound operations like file reading and API calls. Ollama serves as our local large language model provider, eliminating the need for external API dependencies while maintaining privacy for sensitive documents. ChromaDB functions as our vector database, storing document embeddings and enabling semantic search across the document collection. Finally, Quartz provides the web interface for users to interact with the system, upload documents, and engage in conversations about their content.

Asynchronous programming is essential for building responsive applications, especially when dealing with I/O-bound tasks like reading files and making API calls. By introducing async functions from the start, this implementation helps novices build good habits while creating more efficient applications. The async/await syntax in Python makes it relatively straightforward to write non-blocking code without dealing with complex callback structures.

## Setting Up the Environment

Before diving into code, we need to set up our development environment. We'll create a virtual environment to isolate our project dependencies and then install the required packages. This approach ensures that our project won't interfere with other Python projects on the same machine.

### Creating a Virtual Environment

First, open a terminal or command prompt and navigate to your project directory. Run the following commands to create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv resume_rag_venv

# Activate the virtual environment (macOS/Linux)
source resume_rag_venv/bin/activate

# Activate the virtual environment (Windows)
# resume_rag_venv\Scripts\activate
```

Once the virtual environment is activated, your command prompt should show the environment name, indicating that any packages you install will be isolated to this project. The virtual environment creates a sandbox where you can install packages without affecting your system-wide Python installation. This isolation makes it easier to manage dependencies and share your project with others.

### Installing Required Packages

Now, install the necessary packages for our RAG system:

```bash
pip install pypdf chromadb httpx quart aiofiles
```

These packages serve different purposes in our system: pypdf handles PDF document parsing, provides utilities for working with language models, chromadb serves as our vector database, httpx enables asynchronous HTTP requests, Quart provides an asynchronous web framework, and aiofiles allows for asynchronous file operations. This combination of packages gives us everything we need to build a complete RAG system with asynchronous capabilities.

## Document Processing Pipeline

The first step in building our RAG system is creating a document processing pipeline that can extract text from PDFs and prepare it for storage in our vector database. This involves loading PDFs, extracting text, and splitting the text into manageable chunks.

### Asynchronous PDF Processing

Here's how we can implement asynchronous PDF processing:

```python
import asyncio
import os
from pypdf import PdfReader
import aiofiles

async def extract_text_from_pdf(pdf_path):
    """Asynchronously extract text from a PDF file."""
    # PDF reading is CPU-bound, so we run it in a thread pool
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
    
    # Create tasks for processing each PDF
    tasks = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_files]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Create a dictionary mapping filenames to their extracted text
    document_texts = {os.path.basename(path): text for text, path in results}
    return document_texts
```

This code handles PDF processing in an asynchronous manner. The `extract_text_from_pdf` function uses `run_in_executor` to run the synchronous PDF reading operation in a separate thread, preventing it from blocking the event loop. The `process_pdf_directory` function creates tasks for processing each PDF in the directory and executes them concurrently using `asyncio.gather`, significantly speeding up processing when dealing with multiple documents.

### Text Chunking for Vector Storage

After extracting text from PDFs, we need to split it into smaller chunks for effective retrieval. Here's how we can implement text chunking:

```python
async def split_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks asynchronously."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Avoid cutting words by finding the last space before the end
        if end < text_length and text[end] != ' ':
            # Find the last space in the chunk
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
```

The `split_text` function divides text into overlapping chunks of a specified size, being careful not to cut words in the middle. The overlap helps maintain context when retrieving chunks later. The `process_documents` function applies chunking to all documents and returns a list of tuples containing filename, chunk index, and chunk text, which will be useful for tracking the source of each chunk.

## Vector Database with ChromaDB

Now that we have our document chunks, we need to store them in a vector database for efficient retrieval. ChromaDB is a great choice for this purpose as it's simple to use and provides good performance for semantic search applications.

### Initializing ChromaDB

Here's how we can set up ChromaDB asynchronously:

```python
import chromadb
from chromadb.config import Settings

async def setup_chroma():
    """Set up ChromaDB asynchronously."""
    # ChromaDB operations are synchronous, so we run them in a thread pool
    loop = asyncio.get_event_loop()
    client = await loop.run_in_executor(
        None, 
        lambda: chromadb.Client(Settings(persist_directory="./chroma_db"))
    )

    # Create or get the collection
    collection = await loop.run_in_executor(
        None,
        lambda: client.get_or_create_collection("document_chunks")
    )

    return client, collection
```

Since ChromaDB doesn't have native async support, we use `run_in_executor` to run ChromaDB operations in a thread pool, preventing them from blocking the event loop. This function initializes the ChromaDB client and creates a collection for storing our document chunks.

### Adding Documents to ChromaDB

Now let's add our document chunks to ChromaDB:

```python
async def add_chunks_to_chroma(collection, chunks):
    """Add document chunks to ChromaDB asynchronously."""
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": idx} for filename, idx, _ in chunks]
    documents = [chunk for _, _, chunk in chunks]
    
    # ChromaDB operations are synchronous, so we run them in a thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )
    )
```

This function adds our document chunks to ChromaDB, along with metadata about the source document and chunk index. The metadata will be useful later when retrieving chunks and generating responses, as it allows us to trace back to the original document.

## Integrating Ollama for Text Generation

With our documents stored in ChromaDB, we now need to set up Ollama for text generation. Ollama is a tool for running large language models locally, which is perfect for our RAG system as it eliminates the need for external API dependencies.

### Setting Up Ollama Integration

Here's how we can interact with Ollama asynchronously:

```python
import httpx

async def query_ollama(prompt, model="llama2"):
    """Query Ollama API asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "")
```

This function sends a prompt to Ollama's API and returns the generated response. We're using httpx, an asynchronous HTTP client, to make the request without blocking the event loop. The default model is "llama2", but you can change it to any model supported by Ollama.

### RAG Query Processing

Now let's implement the core RAG functionality, which retrieves relevant chunks from ChromaDB and uses them to generate a response with Ollama:

```python
async def process_rag_query(query, collection, model="llama2"):
    """Process a RAG query asynchronously."""
    # Query ChromaDB for relevant chunks
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: collection.query(
            query_texts=[query],
            n_results=3
        )
    )
    
    # Extract the retrieved chunks and their metadata
    retrieved_chunks = results["documents"][0]
    chunk_sources = [meta["source"] for meta in results["metadatas"][0]]
    
    # Prepare the prompt for Ollama
    prompt = f"""
    Answer the following question based on the provided context.
    If you don't know the answer from the context, say "I don't have enough information to answer that."
    
    Context:
    {' '.join(retrieved_chunks)}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate a response using Ollama
    response = await query_ollama(prompt, model)
    
    # Return the response and sources
    return {
        "response": response,
        "sources": chunk_sources
    }
```

This function combines retrieval and generation to implement the RAG pattern. It first queries ChromaDB to find chunks relevant to the user's question, then constructs a prompt that includes these chunks and the original question, and finally sends this prompt to Ollama to generate a response.

## Building the Web UI with Quartz

Finally, we need to create a web UI that allows users to upload documents and chat with them. We'll use Quartz for this purpose, which is a web UI framework that can integrate with our RAG system.

### Basic Web UI Implementation

Let's implement a simple web UI with Quartz:

```python
from quart import Quart, request, jsonify, render_template
import os
import aiofiles

app = Quart(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB
client = None
collection = None

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
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = await request.files
    uploaded_files = files.getlist('file')
    
    for file in uploaded_files:
        if file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            await file.save(file_path)
    
    # Process the uploaded PDFs
    doc_texts = await process_pdf_directory(app.config['UPLOAD_FOLDER'])
    chunks = await process_documents(doc_texts)
    
    # Add chunks to ChromaDB
    await add_chunks_to_chroma(collection, chunks)
    
    return jsonify({"message": "Files uploaded and processed successfully"})

@app.route('/chat', methods=['POST'])
async def chat():
    """Handle chat requests."""
    data = await request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Process the query using RAG
    result = await process_rag_query(query, collection)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

This code sets up a simple web server with routes for the home page, file upload, and chat functionality. The `setup` function initializes ChromaDB before the server starts serving requests. The `upload_file` function handles file uploads, processes the PDFs, and adds them to ChromaDB. The `chat` function processes chat requests using our RAG query processing function.

### HTML Template for the UI

To complete our web UI, we need to create an HTML template:

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Simple RAG Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-container {
            border: 1px solid #ccc;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .ai-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .sources {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .input-container {
            display: flex;
        }
        #user-input {
            flex-grow: 1;
            padding: 10px;
            margin-right: 10px;
        }
        button {
            padding: 10px 20px;
        }
        .upload-container {
            margin-top: 30px;
            padding: 20px;
            border: 1px dashed #ccc;
        }
    </style>
</head>
<body>
    <h1>Simple RAG Chat</h1>
    
    <div class="chat-container" id="chat-container"></div>
    
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Ask a question...">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <div class="upload-container">
        <h2>Upload Documents</h2>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" accept=".pdf" multiple>
            <button type="submit">Upload</button>
        </form>
    </div>
    
    <script>
        // Handle file upload
        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            alert(result.message || result.error);
        });
        
        // Handle chat
        async function sendMessage() {
            const userInput = document.getElementById('user-input');
            const chatContainer = document.getElementById('chat-container');
            
            const message = userInput.value.trim();
            if (!message) return;
            
            // Display user message
            const userMessageElement = document.createElement('div');
            userMessageElement.className = 'user-message';
            userMessageElement.textContent = message;
            chatContainer.appendChild(userMessageElement);
            
            // Clear input
            userInput.value = '';
            
            // Display loading message
            const loadingElement = document.createElement('div');
            loadingElement.className = 'ai-message';
            loadingElement.textContent = 'Thinking...';
            chatContainer.appendChild(loadingElement);
            
            try {
                // Send chat request
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: message })
                });
                
                const result = await response.json();
                
                // Remove loading message
                chatContainer.removeChild(loadingElement);
                
                // Display AI response
                const aiMessageElement = document.createElement('div');
                aiMessageElement.className = 'ai-message';
                aiMessageElement.textContent = result.response;
                
                // Display sources if available
                if (result.sources && result.sources.length > 0) {
                    const sourcesElement = document.createElement('div');
                    sourcesElement.className = 'sources';
                    sourcesElement.textContent = 'Sources: ' + result.sources.join(', ');
                    aiMessageElement.appendChild(sourcesElement);
                }
                
                chatContainer.appendChild(aiMessageElement);
            } catch (error) {
                console.error('Error:', error);
                chatContainer.removeChild(loadingElement);
                
                const errorElement = document.createElement('div');
                errorElement.className = 'ai-message';
                errorElement.textContent = 'Error: Could not get a response.';
                chatContainer.appendChild(errorElement);
            }
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Allow sending message with Enter key
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

This HTML template provides a simple chat interface with a message history, an input field for asking questions, and a file upload section for adding documents to the system. The JavaScript code handles sending chat requests and uploading files asynchronously.

## Putting It All Together

Now let's combine all the pieces into a complete application. Here's the full code for our RAG system:

```python
# app.py
import asyncio
import os
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import httpx
from quart import Quart, request, jsonify, render_template
import aiofiles

# Initialize Quart app
app = Quart(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
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
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "")

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
    
    response = await query_ollama(prompt, model)
    
    return {
        "response": response,
        "sources": chunk_sources
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
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = await request.files
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
    app.run(debug=True)
```

This complete application combines all the components we've built: document processing, vector storage, text generation, and a web UI. It provides a simple but functional RAG system that novices can use as a starting point for their own projects.

## Conclusion

This guide has walked through building a simple but functional RAG tool using Python, Ollama, ChromaDB, and Quartz. By following asynchronous patterns from the start, we've created an efficient application that can handle document processing, vector storage, and text generation without blocking the main thread. This approach is particularly important for web applications, where responsiveness is crucial for a good user experience.

The resulting RAG system allows users to upload PDF documents, ask questions about them, and receive responses that are grounded in the content of those documents. It demonstrates the power of combining retrieval and generation to create more accurate and reliable AI systems. While this implementation is deliberately kept simple for novice AI developers, it provides a solid foundation that can be extended with additional features such as better text splitting, more sophisticated prompt engineering, or integration with different document types.

For next steps, developers could enhance this system by adding authentication, improving the document processing pipeline, implementing document-level filtering, or exploring different embedding models for better semantic search. The asynchronous patterns introduced here will scale well as the application grows in complexity, providing a valuable learning experience for novice AI developers.

