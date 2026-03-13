import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import data_processing

def build_vector_store():
    load_dotenv()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    laws_dir = os.path.join(current_dir, "Laws")
    db_dir = os.path.join(current_dir, "chroma_db")
    
    print(f"Loading and splitting documents from {laws_dir}...")
    docs_generator = data_processing.load_and_split_documents(laws_dir)
    
    docs = list(docs_generator)
    print(f"Total documents loaded: {len(docs)}")
    
    if len(docs) == 0:
        print("No documents found. Please check data_processing.py and Laws directory.")
        return

    print(f"Building vector store and persisting to {db_dir}...")
    
    # 使用 Ollama 的 qwen3-embedding:4b 模型
    embedding_model = OllamaEmbeddings(model="qwen3-embedding:4b")
    print("Using Ollama embedding model: qwen3-embedding:4b")
    
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=db_dir
    )
    
    print("✓ Vector store successfully built and persisted!")

if __name__ == "__main__":
    build_vector_store()
