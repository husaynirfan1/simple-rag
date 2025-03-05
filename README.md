# simple-rag
Simple RAG system utilizing Milvus(Zilliz Cloud) for vector database.

This project use scraped Almanar news for sample data.

## Features
- Use Lingmesscoref to resolved chat history
- Multi-turn. Customisable number of windows turn.
- Including scraping script for Al-Manar English.
- Utilized Milvus and Zilliz Cloud, scaling make easy.

## Installations
1. Build your Milvus database or easier, just use Zilliz Cloud. Build the schema like this in the inserDataChunks.py. This is just for reference as I already commented it.
```
fields = [
#     FieldSchema(
#         name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
#     ),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Store original text
#     FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # Sparse vector field
#     FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),  # Dense vector field with 768 dimensions
# ]
```

2.  Install all requirements.
   ```
pip install -r requirements.txt
  ```
3.  Install Ollama and pull model. In this project, for contextual chunk, the project utilized Lllama3.1 8B and Qwen2.5 14B for interaction with user. You can easily change that in insertDataChunks.py and streamlit app script.
4.  Install NLTK too (chunks were build using NLTK). After chunking, the project inject context to the chunks, so uploaded vectorised hybrid (sparse and dense) will be original data, while text will be contextual chunks.
5.  Run streamlit app.

### Flowchart


### Nota Kaki 
You will see handle_message twice in the streamlit app because the project initially utilized Telegram bot for interface. (there is also pre-write telegram bot code if you want to use telegram bot.)


