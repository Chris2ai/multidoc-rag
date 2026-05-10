"""
检索模块
负责将文档加载、分块、向量化后存入 ChromaDB，并提供检索器接口。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from langchain_chroma import Chroma

from src.loader import load_document
from src.splitter import split_documents
from src.embedder import get_embedder

# 向量库存储路径
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "multidoc_rag"


def build_vectorstore(file_paths: list, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    从文件列表构建向量数据库。
    步骤：逐个文件加载 → 统一分块 → 向量化 → 存入 ChromaDB
    """
    all_chunks = []
    for path in file_paths:
        print(f"📄 正在处理: {path}")
        docs = load_document(path)
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("没有生成任何文档块，请检查文件是否有效")

    embedder = get_embedder()

    # 删除已有的同名 collection，避免数据混杂（生产环境可改为增量更新）
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  已清空旧向量库: {COLLECTION_NAME}")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedder,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"✅ 向量库构建完成：{len(all_chunks)} 个块已存储到 {CHROMA_PATH}")
    return vectorstore


def get_retriever(file_paths: list = None, k: int = 4):
    """
    返回一个检索器实例。
    如果 file_paths 不为空，先构建向量库；如果为空，直接加载已有向量库。
    k 是检索返回的最相似块数量。
    """
    embedder = get_embedder()

    if file_paths:
        vectorstore = build_vectorstore(file_paths)
    else:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=CHROMA_PATH,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


# 测试代码
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python src/retriever.py data/<文件1> [data/<文件2> ...]")
        sys.exit(1)

    files = sys.argv[1:]
    retriever = get_retriever(file_paths=files)

    # 测试检索
    query = "什么是AI售前"
    print(f"\n🔍 测试检索: {query}")
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(doc.page_content[:200])