"""
MultiDoc-RAG Web 界面 (优化版)
增强反馈、错误处理、清空对话功能。
"""
import sys
import os
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.chain import ask_stream
from src.retriever import build_vectorstore

# ---------- 页面配置 ----------
st.set_page_config(page_title="MultiDoc-RAG", page_icon="📚", layout="wide")
st.title("📚 MultiDoc-RAG — 多文档智能问答系统")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 侧边栏：文档管理 ----------
with st.sidebar:
    st.header("📂 文档管理")
    uploaded_files = st.file_uploader(
        "上传 PDF / Word / TXT 文件 (多个)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    # 处理按钮
    if st.button("处理文档", type="primary"):
        if not uploaded_files:
            st.warning("请先上传至少一个文档。")
        else:
            # 保存文件到临时目录
            save_dir = tempfile.mkdtemp()
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(save_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            # 带进度条的构建过程
            progress_bar = st.progress(0, text="正在解析文档...")
            try:
                # 注意：build_vectorstore 内部会打印详细 log，这里简化为进度更新
                progress_bar.progress(30, text="正在分块与向量化...")
                vectorstore = build_vectorstore(file_paths)

                progress_bar.progress(100, text="✅ 向量库构建完成！")
                count = vectorstore._collection.count()
                st.success(f"已成功索引 {len(uploaded_files)} 个文件，共 {count} 个文档块。")

            except ValueError as ve:
                st.error(f"文档处理错误: {ve}")
            except Exception as e:
                error_msg = str(e)
                if "DASHSCOPE_API_KEY" in error_msg or "api_key" in error_msg.lower():
                    st.error("API Key 未配置，请检查项目根目录下的 .env 文件。")
                else:
                    st.error(f"构建向量库失败: {error_msg}")
            finally:
                # 清理进度条
                progress_bar.empty()

    st.divider()
    st.caption("💡 提示：处理完成后，即可在右侧对话窗口中提问。")

    # 清空对话按钮
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# ---------- 主区域：对话窗口 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("💬 请输入你的问题..."):
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 助手回答（流式）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in ask_stream(prompt, session_id="streamlit_user"):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "DASHSCOPE" in error_msg:
                message_placeholder.error("API Key 无效或未设置。")
            else:
                message_placeholder.error(f"生成回答时出错: {error_msg}")
            full_response = ""

    # 只有成功时才记录助手的回答
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})