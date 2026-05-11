"""
检索验证脚本
- 测试多个问题，打印 Top-K 检索结果
- 实验不同 chunk_size 对检索效果的影响
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.retriever import get_retriever

# 测试问题列表（针对简历文档，可自行增删）
TEST_QUESTIONS = [
    "阳玉龙在哪些公司工作过？",
    "他有哪些AI相关的技能或认证？",
    "他的英语水平怎么样？",
    "他在NCH灯饰担任什么职位？",
    "AI售前的核心能力有哪些？",
]

# 要对比的 chunk_size 参数
CHUNK_SIZES = [500, 800, 1000]


def test_retrieval(chunk_size: int, k: int = 4):
    """用指定 chunk_size 构建向量库，然后对测试问题逐个检索"""
    files = [
        "data/测试.txt",
        "data/测试.docx",
        "data/阳玉龙_AI售前.pdf",
    ]

    print(f"\n{'='*50}")
    print(f"🧪 chunk_size = {chunk_size}")
    print(f"{'='*50}")

    # 构建/重建向量库（传入 chunk_size）
    retriever = get_retriever(file_paths=files, chunk_size=chunk_size)

    for question in TEST_QUESTIONS:
        print(f"\n❓ 问题: {question}")
        results = retriever.invoke(question)
        for i, doc in enumerate(results[:k]):
            snippet = doc.page_content.replace("\n", " ")[:150]
            print(f"  📄 结果{i+1}: {snippet}...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # 运行所有 chunk_size 对比
        for cs in CHUNK_SIZES:
            test_retrieval(chunk_size=cs)
    else:
        # 默认只测试 chunk_size=500
        test_retrieval(chunk_size=500)