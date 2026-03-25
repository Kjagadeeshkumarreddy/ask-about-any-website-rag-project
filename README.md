# 🌐 Web RAG - Ask Any Question About a Website

This project is a **Streamlit application** that lets you paste a website URL and ask questions about its content.  
It uses **LangChain**, **Ollama embeddings**, and **FAISS vector store** to perform Retrieval-Augmented Generation (RAG).

---

## 🛠️ Tech Stack
- **Streamlit** – UI framework
- **LangChain** – RAG pipeline
- **Ollama** – Embeddings & LLM
- **FAISS** – Vector store
- **Python** – Core language

---

## 📦 Installation & Virtual Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kjagadeeshkumarreddy/ask-about-any-website-rag-project.git
cd ask-about-any-website-rag-project

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Make sure you have Ollama installed and pull required models
ollama pull mxbai-embed-large
ollama pull gemma3:1b
