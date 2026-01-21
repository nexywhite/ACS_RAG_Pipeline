import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# LOAD VECTOR DB
CHROMA_DIR = "chroma_db"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# RETRIEVAL
def retrieve_chunks(query, k=3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


# PROMPT TEMPLATE
def build_prompt(question, retrieved_docs):
    context = ""

    for i, (doc, score) in enumerate(retrieved_docs):
        context += f"[{i}] Source: {doc.metadata.get('source', '')}\n"
        context += doc.page_content + "\n\n"

    prompt = f"""
Use ONLY the information in the context to answer the question.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
"""
    return prompt


# OLLAMA MODEL
def get_llm():
    return ChatOllama(model="llama3.2:1b", temperature=0.2)   # Change to any model you like


# STREAMLIT UI
def main():
    st.title("RAG Chatbot (PDF + Text + Image OCR)")
    st.write("Ask a question and compare Base LLM vs RAG results.")

    question = st.text_input("Enter your question:")
    k = st.slider("Top-k retrieved chunks", 1, 10, 3)

    if st.button("Run"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        llm = get_llm()

        # Base Answer
        st.subheader("Base LLM Answer (no retrieval)")
        base_response = llm.invoke(question)
        st.write(base_response.content)

        # RAG Answer
        st.subheader("RAG Answer (with retrieval)")
        retrieved = retrieve_chunks(question, k=k)
        rag_prompt = build_prompt(question, retrieved)
        rag_response = llm.invoke(rag_prompt)
        st.write(rag_response.content)

        # Retrieved Chunks
        st.subheader("Retrieved Chunks")
        for i, (doc, score) in enumerate(retrieved):
            with st.expander(f"Chunk {i} — source: {doc.metadata.get('source')}"):
                st.write(f"Score: {score}")
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
