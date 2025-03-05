import os
import numpy as np
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
)
from pymilvus import MilvusClient
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from ollama import chat
from ollama import ChatResponse
import re 
import asyncio

# Get the current date as a string in 'YYYY-MM-DD' form
current_date_str = datetime.now().strftime("%Y-%m-%d")

# Connect to Zilliz Cloud (Milvus)
client = MilvusClient(
    uri="https://in03-e258b56feec6af7.serverless.gcp-us-west1.cloud.zilliz.com",
    token="927488e836671a8d541cf0c2d939f9338aaa3e44828c044c56e46db941054a6906b4ffee7d3cf3eb3fa44446a802d488d4fdae97"
)
# Connect to Milvus server (Zilliz Cloud)
connections.connect(
    alias="default", 
    uri="https://in03-e258b56feec6af7.serverless.gcp-us-west1.cloud.zilliz.com", 
    token="927488e836671a8d541cf0c2d939f9338aaa3e44828c044c56e46db941054a6906b4ffee7d3cf3eb3fa44446a802d488d4fdae97"
)

# Download NLTK punkt tokenizer (run once if not already downloaded)
nltk.download('punkt_tab')        
def chunk_news_article_nltk(text, max_chars=1000, overlap=100):
    """
    Chunk news article text into sentences using NLTK, with a max character limit and overlap.
    Args:
        text (str): Input news article text.
        max_chars (int): Max characters per chunk (default 1000, ~250 tokens for BGE-M3).
        overlap (int): Characters to overlap between chunks (default 100).
    Returns:
        list: List of chunks preserving sentence-level information.
    """
    # Tokenize text into sentences using NLTK
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        # Check if adding the sentence exceeds max_chars
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            # If current_chunk has content, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Apply overlap by taking the last overlap chars
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_buffer = current_chunk[-overlap:]
                else:
                    overlap_buffer = ""
            
            # Start new chunk with overlap and current sentence
            current_chunk = overlap_buffer + sentence + " "
            
            # Handle case where a single sentence exceeds max_chars
            if len(current_chunk) > max_chars:
                # Truncate to max_chars and find last sentence boundary within limit
                truncated = current_chunk[:max_chars]
                last_sentence_end = truncated.rfind('. ')  # Look for sentence end
                if last_sentence_end != -1 and last_sentence_end > overlap:
                    current_chunk = truncated[:last_sentence_end + 1]
                    chunks.append(current_chunk.strip())
                    # Reset with remainder as overlap buffer
                    overlap_buffer = truncated[last_sentence_end + 1:]
                    current_chunk = overlap_buffer
                else:
                    # No sentence boundary found, cut at max_chars
                    chunks.append(truncated.strip())
                    overlap_buffer = current_chunk[max_chars - overlap:max_chars] if overlap > 0 else ""
                    current_chunk = overlap_buffer
    
    # Add the final chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Load and chunk news articles
def situate_context(doc: str, chunk: str):
    DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and include date, title and url if exist within overall document in your answer. Do not add anything else.
    
    """
    
    ctx = f"""
    Document: {doc}

    Chunk: {chunk}
    
    If available in the document, extract the following metadata:
    * URL: (If a URL is present, include it here)
    * Date and Time: (If a date and time are present, include them here)
    
    Then, using only the information from the provided document, create a contextual explanation of the chunk. If possible, include direct quotes from the document that support your explanation.
    
    Output format:
    
    Context: [Contextual Explanation]
    URL: [Extracted URL or 'N/A']
    Date and Time: [Extracted Date/Time or 'N/A']
    """

    formatted_docc = DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)
    formatted_chunk = CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)

    textual = f"{formatted_docc}\n{formatted_chunk}"

    response: ChatResponse = chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": ctx}]
    )

    #------------------------------------------------------------------------
    # Sample string with custom tags
    text = response["message"]["content"]

    # Regular expression to match the custom tags and extract their content
    tags = re.findall(r'<document>(.*?)</document>', text)

    # if not response.content or not hasattr(response.content[0], 'text'):
    #     raise ValueError(f"Invalid response structure: {response}")
        
    return text
    
async def process_documents(folder_path: str):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs.append(file.read())

    resolved = []
    original_chunk = []
    
    for doc in docs:
        chunks = chunk_news_article_nltk(doc, max_chars=1000, overlap=100)  # Assuming this exists
        for chunk in chunks:
            context = situate_context(doc, chunk)
            resolved.append(context)
            original_chunk.append(chunk)
            print(f"Context: {context}")
            
    # Preview results
    print(f"Total chunks: {len(resolved)}")
    for i, chunk in enumerate(resolved[:3]):
        print(f"Chunk {i} ({len(chunk)} chars): {chunk}")
    return resolved, original_chunk

async def main():
    folder_path = input("Path: ")
    resolved, original_chunk = await process_documents(folder_path)
    print(f"Processed {len(resolved)} chunks with context.")
    return resolved, original_chunk

# Initialize BGEM3EmbeddingFunction for generating dense and sparse vectors
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
# dense_dim = embedding_function.dim["dense"]

# # Embed the documents using BGEM3 (or random embedding for testing)
# chunks_embeddings = embedding_function(resolved)  # This should return the dense and sparse vectors

# # Define the schema for the collection
# fields = [
#     FieldSchema(
#         name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
#     ),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Store original text
#     FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # Sparse vector field
#     FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),  # Dense vector field with 768 dimensions
# ]

# # Create the collection schema
# schema = CollectionSchema(fields, description="Collection with dense and sparse vectors")

# # Create indices for the vector fields
# sparse_index = {
#     "index_type": "SPARSE_INVERTED_INDEX",  # Sparse vector index type
#     "metric_type": "IP",  # Similarity metric (Inner Product)
# }
# col.create_index("sparse_vector", sparse_index)

# dense_index = {
#     "index_type": "FLAT",  # Dense vector index type
#     "metric_type": "IP",  # Similarity metric (Inner Product)
# }
# col.create_index("dense_vector", dense_index)

# # Load the collection into memory to allow searching
# col.load()

def insert_data(resolved, chunks_embeddings, col, col_name):
    # Insert the document text and their corresponding embeddings into Milvus
    batch_size = 50  # Size of each insertion batch
    for i in range(0, len(resolved), batch_size):
        batched_entities = [
            resolved[i:i + batch_size],  # Document texts (Auto_id assumed)
            chunks_embeddings["sparse"][i:i + batch_size],  # Sparse vectors
            chunks_embeddings["dense"][i:i + batch_size],  # Dense vectors
        ]
        col.insert(batched_entities)  # Insert the batched data

    # Flush to persist to storage
    col.flush()

    print(f"Collection '{col_name}' created and data inserted successfully!")

if __name__ == "__main__":
    # Initialize BGEM3EmbeddingFunction
    embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = embedding_function.dim["dense"]
    
    # Run async main to get resolved
    resolved, original_chunk = asyncio.run(main())

    # Embed the resolved chunks
    chunks_embeddings = embedding_function(original_chunk)  # Generate dense and sparse vectors for raw chunks
    # Assuming col and col_name are defined
    col_name = "news_mideast"
    col = Collection(col_name)  # Adjust schema/connection as needed
    insert_data(resolved, chunks_embeddings, col, col_name)