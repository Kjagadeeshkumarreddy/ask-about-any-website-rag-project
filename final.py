import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Web RAG", layout="wide")
st.title("🌐 ASK ANY QUESTION ABOUT WEBSITE")

url = st.text_input("Paste Website URL:", placeholder="https://example.com")

@st.cache_resource
def get_vector_store(url):
    if not url:
        return None
    loader = WebBaseLoader(url)
    data = loader.load()
    content = data[0].page_content
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks_data = splitter.split_text(content)
    print(chunks_data[:10])
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISS.from_texts(chunks_data, embedding=embeddings)
    return vector_store


if url:
    with st.spinner("Indexing website..."):
        vs = get_vector_store(url)
    
    if vs:
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Generating answer..."):
                docs_with_scores = vs.similarity_search_with_score(question, k=2)
                
                best_ans_text = docs_with_scores[0][0].page_content
                
                llm = ChatOllama(model="gemma3:1b", temperature=0)
                
                template = """
                Answer the question based ONLY on the following context:
                {context}
                
                Question: {question}"""
                
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm
                
                final_ans = chain.invoke({"context": best_ans_text, "question": question})
                
                st.subheader("Answer:")
                st.write(final_ans.content)
                
                with st.expander("Show source context"):
                    st.info(best_ans_text)
else:
    st.info("Please enter a URL to begin.")