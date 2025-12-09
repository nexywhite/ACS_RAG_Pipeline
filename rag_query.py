import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"


# LOAD VECTOR DATABASE
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )
    return vectorstore


# RETRIEVE CHUNKS
def retrieve_chunks(query, k=3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    # results = [(Document, score), ...]
    return results


# BUILD RAG PROMPT
def build_rag_prompt(question, retrieved):
    context = ""

    for i, (doc, score) in enumerate(retrieved):
        context += f"[{i}] Source: {doc.metadata.get('source', '')}\n"
        context += doc.page_content + "\n\n"

    prompt = f"""
Use ONLY the information from the context to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# LOAD OLLAMA MODEL
def get_llm():
    return ChatOllama(model="llama3.1")


# COMPARE BASE VS RAG ANSWER
def ask_question(question):
    llm = get_llm()

    print("\nUSER QUESTION:", question)
    print("-" * 60)

    # Base answer (no retrieval)
    print("\nBase LLM answer (no retrieval):")
    base = llm.invoke(question)
    print(base.content)

    # Retrieval
    retrieved = retrieve_chunks(question, k=3)

    # Build RAG prompt
    rag_prompt = build_rag_prompt(question, retrieved)

    # RAG answer
    print("\nRAG-enhanced answer:")
    rag_answer = llm.invoke(rag_prompt)
    print(rag_answer.content)

    print("\nRetrieved Chunks:")
    for i, (doc, score) in enumerate(retrieved):
        print(f"\n[{i}] (Score: {score}) Source: {doc.metadata.get('source')}")
        print(doc.page_content[:300] + "...")
    


if __name__ == "__main__":
    q = input("Enter your question: ")
    ask_question(q)
