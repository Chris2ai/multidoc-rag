"""
RAG 链组装模块（含对话记忆与流式输出）
重构版本：将检索步骤外置，令 RunnableWithMessageHistory 只管理对话历史。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from dotenv import load_dotenv

from src.retriever import get_retriever

load_dotenv()

# ---------- 1. 检索器 ----------
retriever = get_retriever()

# ---------- 2. 格式化文档块 ----------
def format_docs(docs):
    """格式化检索到的文档块，并加上便于引用的序号。"""
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', '未知')
        content = doc.page_content
        formatted.append(
            f"【块{i} - 来源：{source}】\n{content}"
        )
    return "\n\n".join(formatted)

# ---------- 3. 大模型 ----------
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)

# ---------- 4. 最终的问答提示模板 ----------
# 注意：不再包含 document_context 变量，文档内容会直接拼接到 question 中。
qa_template = """你是一个基于文档知识的问答助手。请严格按照以下规则回答：
1. 优先根据提供的文档内容回答，在答案中引用具体的【块X - 来源：...】作为依据。
2. 如果文档中没有相关信息，请明确说明“文档中未找到相关信息”。
3. 回答时可以参考对话历史，但不要捏造文档以外的信息。
4. 在回答的最后，单独一行列出“参考来源：”，并列出所有用到的文档块（例如“块1, 块2 来自 xxx.pdf”）。"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ---------- 5. 基础的问答链（不含历史包装） ----------
base_qa_chain = (
    qa_prompt
    | llm
    | StrOutputParser()
)

# ---------- 6. 会话历史存储 ----------
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ---------- 7. 带历史的问答链 ----------
qa_chain_with_history = RunnableWithMessageHistory(
    base_qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ---------- 8. 完整的 RAG 链 ----------
def build_rag_chain():
    """构建并返回一个先检索、再回答的链。"""
    
    def combine_context_and_question(inputs: dict):
        """从输入中提取上下文和问题，拼成一个新问题。"""
        user_question = inputs["question"]
        retrieved_docs = inputs.get("context_docs", [])
        if retrieved_docs:
            context_str = "文档内容：\n" + format_docs(retrieved_docs)
        else:
            context_str = "文档中未找到相关信息。"
        
        # 将文档上下文和原始问题合并，作为新的“question”传入带历史的链
        new_question = f"{context_str}\n\n用户问题：{user_question}"
        return {"question": new_question}

    rag_chain = (
        RunnableParallel(
            question=itemgetter("question"),
            context_docs=itemgetter("question") | retriever
        )
        | RunnableLambda(combine_context_and_question)
        | qa_chain_with_history
    )
    return rag_chain

# 实例化完整的 RAG 链
full_rag_chain = build_rag_chain()

# ---------- 9. 封装调用函数 ----------
def ask(question: str, session_id: str = "default") -> str:
    """非流式问答。"""
    return full_rag_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

def ask_stream(question: str, session_id: str = "default"):
    """流式问答。"""
    for chunk in full_rag_chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk

# ---------- 10. 测试代码 ----------
if __name__ == "__main__":
    print("🚀 多文档 RAG 对话测试（流式 + 记忆）\n")

    # 第一轮
    q1 = "阳玉龙在哪些公司工作过？"
    print(f"👤 用户: {q1}")
    print("🤖 助手: ", end="", flush=True)
    for chunk in ask_stream(q1):
        print(chunk, end="", flush=True)
    print("\n")

    # 第二轮（验证上下文记忆）
    q2 = "他在这些公司里都担任过什么职位？"
    print(f"👤 用户: {q2}")
    print("🤖 助手: ", end="", flush=True)
    for chunk in ask_stream(q2):
        print(chunk, end="", flush=True)
    print("\n")

    # 第三轮（验证文档检索与记忆的综合）
    q3 = "他有AI相关认证吗？如果有，请告诉我认证名称。"
    print(f"👤 用户: {q3}")
    print("🤖 助手: ", end="", flush=True)
    for chunk in ask_stream(q3):
        print(chunk, end="", flush=True)
    print()