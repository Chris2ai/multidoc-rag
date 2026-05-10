"""
文档加载模块
支持 PDF、Word、TXT 格式，提供统一入口函数 load_document
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_document(file_path: str):
    """
    根据文件扩展名自动选择合适的加载器，返回加载后的文档对象列表。
    每个文档对象包含 page_content 和 metadata。
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    docs = loader.load()
    print(f"✅ 成功加载 {file_path}，共 {len(docs)} 个文档块")
    return docs


# 测试代码（仅在直接运行本文件时执行）
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python src/loader.py <文件路径>")
        sys.exit(1)

    test_path = sys.argv[1]
    docs = load_document(test_path)
    for i, doc in enumerate(docs):
        print(f"\n--- 块 {i+1} ---")
        print(doc.page_content[:200])
        print("...")