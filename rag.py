from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

st.title("🌐 ASK ANY QUESTION ABOUT WEBSITE")
url=st.text_input("Paste Website URL:", placeholder="https://example.com")

loader=WebBaseLoader(url)
data=loader.load()
data=data[0].page_content
    
#data chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks_data=splitter.split_text(data)

#embidings
embeddings =OllamaEmbeddings(model="mxbai-embed-large")

#store in vector store
vector_store=FAISS.from_texts(chunks_data,embedding=embeddings)

# st.title("enter the question")
question=st.text_input("enter the question")
answer=vector_store.similarity_search_with_score(question)

best_score=answer[0][1]
best_ans=answer[0][0].page_content

llm=ChatOllama(model="gemma3:1b",temperature=0)


template='''
    Answer the question based ONLY on the following context:
    {context}
    Question: {question}
'''
prompt=ChatPromptTemplate.from_template(template)


chain=prompt|llm
final_ans=chain.invoke({"context":best_ans,"question":question})

st.markdown(final_ans.content)