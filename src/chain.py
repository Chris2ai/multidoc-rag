"""
RAG 链组装模块（含对话记忆、流式输出、引用溯源）
Streamlit 兼容版：每次调用动态加载最新的向量库。
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

# ---------- 延迟加载检索器 ----------
def _get_retriever():
    """每次调用时重新加载，确保获取到最新的向量库。"""
    return get_retriever()

# ---------- 格式化文档块（加入序号） ----------
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

# ---------- 大模型 ----------
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)

# ---------- 提示模板（要求引用来源） ----------
qa_template = """你是一个基于文档知识的问答助手。请严格按照以下规则回答：
1. 优先根据提供的文档内容回答，在答案中引用具体的【块X - 来源：...】作为依据。
2. 如果文档中没有相关信息，请明确说明"文档中未找到相关信息"。
3. 回答时可以参考对话历史，但不要捏造文档以外的信息。
4. 在回答的最后，单独一行列出"参考来源："，并列出所有用到的文档块（例如"块1, 块2 来自 xxx.pdf"）。"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ---------- 基础问答链 ----------
base_qa_chain = (
    qa_prompt
    | llm
    | StrOutputParser()
)

# ---------- 会话历史存储 ----------
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ---------- 带历史的问答链 ----------
qa_chain_with_history = RunnableWithMessageHistory(
    base_qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ---------- 完整 RAG 链 ----------
def build_rag_chain():
    """构建 RAG 链——每次调用时动态加载最新向量库。"""

    def retrieve_docs(inputs: dict):
        """从输入字典中提取问题，检索相关文档块。"""
        question = inputs["question"]
        r = _get_retriever()
        return r.invoke(question)

    def combine_context_and_question(inputs: dict):
        user_question = inputs["question"]
        retrieved_docs = inputs.get("context_docs", [])
        if retrieved_docs:
            context_str = "文档内容：\n" + format_docs(retrieved_docs)
        else:
            context_str = "文档中未找到相关信息。"

        new_question = f"{context_str}\n\n用户问题：{user_question}"
        return {"question": new_question}

    rag_chain = (
        RunnableParallel(
            question=itemgetter("question"),
            context_docs=RunnableLambda(retrieve_docs)
        )
        | RunnableLambda(combine_context_and_question)
        | qa_chain_with_history
    )
    return rag_chain

full_rag_chain = build_rag_chain()

# ---------- 调用接口 ----------
def ask(question: str, session_id: str = "default") -> str:
    return full_rag_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

def ask_stream(question: str, session_id: str = "default"):
    for chunk in full_rag_chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk

# ---------- 测试 ----------
if __name__ == "__main__":
    print("🚀 多文档 RAG 对话测试（引用溯源）\n")

    q1 = "阳玉龙在哪些公司工作过？"
    print(f"👤 用户: {q1}")
    print("🤖 助手: ", end="", flush=True)
    for chunk in ask_stream(q1):
        print(chunk, end="", flush=True)
    print("\n")

    q2 = "他有AI相关认证吗？"
    print(f"👤 用户: {q2}")
    print("🤖 助手: ", end="", flush=True)
    for chunk in ask_stream(q2):
        print(chunk, end="", flush=True)
    print()