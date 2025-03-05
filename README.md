# Simple RAG 📌

*A not-so-lightweight Retrieval-Augmented Generation (RAG) system utilizing Milvus (Zilliz Cloud) as a vector database.*

This project uses scraped **Al-Manar News** as sample data.

## 🚀 Features
- **Coreference Resolution**: Uses `lingmesscoref` to resolve chat history.
- **Multi-turn Conversation**: Customizable number of window turns.
- **Web Scraping**: Includes a scraping script for Al-Manar English.
- **Scalability**: Utilizes **Milvus** and **Zilliz Cloud**, making scaling easy.

## 🛠 Installation

### 1️⃣ Set Up Milvus (Zilliz Cloud Recommended)
You can either build your own Milvus database or use **Zilliz Cloud** for convenience.
The schema in `insertDataChunks.py` should look like this (for reference, it is already commented in the script):

```python
fields = [
    # FieldSchema(
    #     name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    # ),
    # FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Store original text
    # FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # Sparse vector field
    # FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),  # Dense vector field with 768 dimensions
    # )
]
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Install & Configure Ollama
This project uses **Llama 3.1 8B** for contextual chunking and **Qwen2.5 14B** for user interaction. You can modify these in `insertDataChunks.py` and the Streamlit app script.

### 4️⃣ Install NLTK
This project uses **NLTK** for text chunking. After chunking, context is injected into the chunks, ensuring that the uploaded vectorized hybrid (sparse and dense) data includes both the original text and contextual chunks.

```sh
pip install nltk
```

### 4️⃣ Process and Insert Data
Run insertDataChunks.py. It will ask to input path of folder. Use `folder` path in the current repo for testing.

```python
python insertDataChunks.py
Path: /folder
```

### 5️⃣ Run Streamlit App
```sh
streamlit run streamlit_app.py
```

## 🔄 Flowchart

*The project initially utilized DeepSeek R1 for its reasoning capabilities. However, for general use, it is recommended to use a non-reasoning model.*

![Flowchart](https://github.com/husaynirfan1/simple-rag/blob/main/albai_v3.drawio.png)

## 💡 Notes
- You will see **handle_message** twice in the Streamlit app. This is because the project was initially designed for a **Telegram bot** as the interface.
- The repository also includes a **pre-written Telegram bot script** if you want to use it.

---

## 📌 License
This project is open-source under the **MIT License**.

## 🔗 Links
- [Zilliz Cloud](https://zilliz.com/cloud)
- [Milvus](https://milvus.io/)
- [Ollama](https://ollama.com/)
- [NLTK](https://www.nltk.org/)
- [Project Repository](https://github.com/husaynirfan1/simple-rag)

---

## 📧 Contact
For any questions or collaboration, reach out via:
- **Gmail**: [mhusaynirfan@gmail.com](mailto:mhusaynirfan@gmail.com)
- **GitHub Issues**: [Open an issue](https://github.com/husaynirfan1/simple-rag/issues)
- **Linked In**: [Husayn Irfan](https://my.linkedin.com/in/husayn-irfan-7b4103258)
