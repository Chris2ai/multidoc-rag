"""
向量化模块
使用本地 sentence-transformers 模型将文本块转为向量。
备选：如需切换回云端 Embedding API，只需修改此文件。
"""
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder():
    """
    返回一个本地 Embedding 实例。
    默认使用 BGE 中文小模型（384维，速度快，无需 GPU）。
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )


# 快速测试
if __name__ == "__main__":
    embedder = get_embedder()
    test_texts = ["你好，世界", "这是一段测试文本"]
    vectors = embedder.embed_documents(test_texts)
    print(f"✅ 向量化成功：{len(vectors)} 条文本 → 每条维度 {len(vectors[0])}")