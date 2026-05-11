"""
RAG 链组装模块
使用 LangChain Expression Language (LCEL) 将检索、提示模板、模型串联。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from src.retriever import get_retriever

load_dotenv()

# 1. 准备检索器（加载已有的向量库，不重建）
retriever = get_retriever()  # 默认从 chroma_db/ 读取，k=4, chunk_size=800

# 2. 设计提示词模板
template = """你是一个基于文档知识的问答助手。请根据以下文档内容回答用户问题。
如果文档中没有相关信息，请如实说明"文档中未找到相关信息"。

文档内容：
{context}

用户问题：{question}
回答（如有具体来源可注明文档段落）："""

prompt = ChatPromptTemplate.from_template(template)

# 3. 初始化大模型
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,  # 0.3 偏严谨，减少幻觉
)

# 4. 定义格式化函数：将检索到的文档块列表拼成一段上下文字符串
def format_docs(docs):
    return "\n\n".join(
        f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
        for doc in docs
    )

# 5. 使用 LCEL 组装 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 封装一个方便调用的函数
def ask(question: str) -> str:
    """
    向 RAG 系统提问，返回基于文档的答案。
    """
    return rag_chain.invoke(question)


# 测试代码
if __name__ == "__main__":
    test_questions = [
        "阳玉龙在哪些公司工作过？",
        "他有哪些AI相关的认证？",
        "什么是OCR？",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 用户: {q}")
        print(f"{'='*60}")
        answer = ask(q)
        print(f"🤖 助手: {answer}")