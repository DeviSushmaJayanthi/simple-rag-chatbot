# simple-rag-chatbot
# 📄 PDF Question Answering System

A **Retrieval-Augmented Generation (RAG)** application that lets you ask natural language questions about your PDF documents. Powered by **Google Gemini**, **FAISS vector search**, and **LangChain** — with a clean **Streamlit** interface.

---

## 🚀 Features

- 📂 Load and index an entire directory of PDF files automatically
- 🔍 Semantic search using FAISS vector store with Gemini embeddings
- 🤖 Accurate, context-aware answers via `gemini-2.5-flash`
- 📌 Shows source filenames for every answer (full transparency)
- 💾 Persists the vector store locally — no re-processing on subsequent runs
- 🌐 Simple, interactive UI via Streamlit

---

## 🧠 How It Works

```
PDFs → Chunking → Gemini Embeddings → FAISS Index
                                           ↓
User Query → Embedding → Top-K Retrieval → Gemini LLM → Answer + Sources
```

1. **Document Loading** — Scans a directory for PDF files using `PyPDFDirectoryLoader`
2. **Text Splitting** — Splits content into overlapping chunks (1000 chars, 200 overlap)
3. **Embedding & Indexing** — Converts chunks to vectors using `gemini-embedding-001` and stores them in FAISS
4. **Retrieval** — On each query, retrieves the top 5 most relevant chunks
5. **Answer Generation** — Sends retrieved context + query to `gemini-2.5-flash` for a structured response

---

## 🗂️ Project Structure

```
├── app.py               # Main application (loader, RAG chain, Streamlit UI)
├── .env                 # Environment variables (not committed)
├── vectorstore/         # Auto-created: persisted FAISS index
├── data/                # Your PDF files go here (configurable via DATA_PATH)
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
DATA_PATH=./data
DB_FAISS_PATH=./vectorstore/db_faiss
```

> 💡 Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Add Your PDFs

Place your PDF files inside the directory specified by `DATA_PATH` (default: `./data/`).

### 6. Run the App

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
langchain-classic
faiss-cpu
streamlit
python-dotenv
pypdf
```

> Install all at once: `pip install -r requirements.txt`

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. On the **first run**, the app will automatically process your PDFs and build the vector store (this may take a moment depending on file size)
3. Type your question in the text input field
4. Click **"Get Response"**
5. View the answer and the source PDF filenames it referenced

---

## 🔧 Configuration

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key | *(required)* |
| `DATA_PATH` | Directory containing your PDF files | *(required)* |
| `DB_FAISS_PATH` | Path to save/load the FAISS index | *(required)* |
| `chunk_size` | Text chunk size for splitting | `1000` |
| `chunk_overlap` | Overlap between chunks | `200` |
| `temperature` | LLM response creativity (0 = factual) | `0.2` |
| `k` | Number of chunks retrieved per query | `5` |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `gemini-embedding-001` |
| Vector Store | FAISS (local) |
| Orchestration | LangChain |
| PDF Loading | PyPDFDirectoryLoader |
| UI | Streamlit |

---

## ⚠️ Notes

- The FAISS index is built **once** and reused on subsequent runs. To re-index (e.g., after adding new PDFs), delete the `vectorstore/` directory and restart the app.
- `allow_dangerous_deserialization=True` is set for local FAISS loading — only use index files you created yourself.
- The LLM is instructed to respond with **"I don't know"** if the answer isn't found in the provided PDFs, reducing hallucination.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS by Meta](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
