import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载 .env 中的 API Key
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化模型（通义千问通过 OpenAI 兼容接口调用）
llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=api_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
)

# 发送一条测试消息
messages = [HumanMessage(content="用一句话介绍什么是RAG技术")]

# 调用模型并打印回复
response = llm.invoke(messages)
print("✅ 模型回复：")
print(response.content)

# ========== 流式输出测试 ==========
print("\n✅ 流式输出效果：")
for chunk in llm.stream("请用三句话介绍LangChain框架"):
    print(chunk.content, end="", flush=True)
print()