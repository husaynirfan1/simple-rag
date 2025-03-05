# simple-rag
Simple RAG system utilizing Milvus(Zilliz Cloud) for vector database.

Ensure to install NLTK too (chunks were build using NLTK). After chunking, the project inject context to the chunks, so uploaded vectorised hybrid (sparse and dense) will be original data, while text will be contextual chunks.

This project use scraped Almanar news for sample data.
