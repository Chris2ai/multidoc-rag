"""
文本分块模块
将长文档按语义边界切割成指定大小、带重叠的块，提高检索质量。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    对文档列表进行递归字符分割。

    参数:
        docs:        由 loader 加载的文档对象列表
        chunk_size:  每个块的目标最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    返回:
        分割后的文档块列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 分块完成：{len(docs)} 个文档 → {len(chunks)} 个块")
    return chunks


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")          # 临时在测试时让 src 目录下的模块可互相引用
    from loader import load_document

    if len(sys.argv) < 2:
        print("用法: python src/splitter.py data/<文件名>")
        sys.exit(1)

    file_path = sys.argv[1]
    docs = load_document(file_path)
    chunks = split_documents(docs)

    for i, chunk in enumerate(chunks[:5]):  # 仅预览前 5 个块
        print(f"\n--- 块 {i+1} (长度 {len(chunk.page_content)} 字符) ---")
        print(chunk.page_content[:150])
        print("...")