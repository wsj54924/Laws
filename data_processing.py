import os
import sys
from typing import Iterator, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, TextLoader

# 增加 TextLoader 作为 .md 文件的首选或备选，因为它更轻量且不容易出错


def load_documents_from_directory(directory_path: str) -> Iterator[Document]:
    """
    遍历指定目录，根据文件类型加载所有文档。
    支持 .pdf, .docx, .doc, .md 格式。
    返回一个生成器，逐个 yield 文档，节省内存。
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return


    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            loader = None
            try:
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_ext in [".docx", ".doc"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_ext == ".md":
                    loader = TextLoader(file_path, encoding='utf-8')
                if loader:
                    # load() 返回的是一个列表，通常一个文件包含一个或多个 Document
                    docs = loader.load()
                    for doc in docs:
                        # 添加元数据，方便后续追踪来源
                        parts=file_path.split("\\")
                        code_name=""
                        if "Laws" in parts:
                            idx=parts.index("Laws")
                            if idx+1<len(parts):
                                if idx+1<len(parts):
                                    code_name=parts[idx+1]
                        doc.metadata["source"] = code_name
                        doc.metadata["file_path"] = file
                        yield doc         
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

import re
import json

def split_into_articles(doc: Document) -> List[Document]:
    text = doc.page_content
    filename = doc.metadata.get("source", "")
    file_path = doc.metadata.get("file_path", "")
   
    source = filename
    chapter = os.path.splitext(os.path.basename(file_path))[0]

    # 正则表达式匹配“第X条”
    # 考虑到有些法律条文可能在行首，有些可能在段落中
    # 模式：匹配换行符或开头，跟着“第...条”，然后是空格或内容
    pattern = r'(?:\n|^)(第[一二三四五六七八九十百千万]+条)\s+'
    
    parts = re.split(pattern, text)
    
    new_docs = []
    
    # parts[0] 通常是文档的开头（如标题、前言等），如果内容较多也可以保留，但这里按条切分
    # 如果 parts[0] 包含“章”的信息，我们可以尝试提取
    
    current_chapter_info = chapter
    
    for i in range(1, len(parts), 2):
        article_id = parts[i]
        text = parts[i+1].strip()
        content=re.sub(r'\n\n','',text)
        # 进一步清理 content，如果内容中包含了下一章的标题，可能需要处理
        # 但通常 re.split(pattern, text) 已经把条目分开了
        
        # 如果 content 中包含 "### 第...章"，可能需要更新 chapter (如果用户需要更细的粒度)
        # 但根据用户示例，chapter 设为 "合同编" 即可
        
        new_doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "chapter": current_chapter_info,
                "article_id": article_id
            }
        )
        new_docs.append(new_doc)
        
    return new_docs

def load_and_split_documents(directory_path: str) -> Iterator[Document]:
    for doc in load_documents_from_directory(directory_path):
        for chunk in split_into_articles(doc):
            yield chunk
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
embedding_model=OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-4B")
def vector_store(docs):
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever()
    return vector_store,retriever

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    laws_dir = os.path.join(current_dir, "Laws", "经济法") # 先测试民法典
    
    print("--- Testing Document Splitting ---")
    doc_generator = load_and_split_documents(laws_dir)
    
    count = 0
    
    for chunk in doc_generator:
        count += 1
        
        # 打印前 3 个作为参考
        if count <= 5:
            print(f"\nChunk {count} example:")
            print(json.dumps({
                "content": chunk.page_content[:100] + "...",
                "metadata": chunk.metadata
            }, ensure_ascii=False, indent=4))
