# 📚 MultiDoc-RAG — 多文档智能问答系统

基于 LangChain + ChromaDB + 通义千问 的多文档 RAG 智能问答系统。  
支持上传 PDF / Word / TXT 文档，自动构建向量库，通过自然语言提问获得带引用来源的流式回答。

## ✨ 功能特点

- 📂 **多格式文档支持**：PDF、Word (.docx)、TXT 三种格式自动识别与解析
- 🔍 **语义检索**：基于 BGE 中文模型向量化，ChromaDB 持久化存储，支持语义相似度检索
- 💬 **流式对话**：打字机效果的实时回答，支持多轮对话记忆
- 📎 **引用溯源**：每个回答标注信息来源（文档名 + 段落编号）
- 🖥️ **Web 界面**：Streamlit 构建的简洁交互界面，开箱即用

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 框架 | LangChain | RAG 链组装（LCEL） |
| 模型 | 通义千问 (qwen-turbo) | 答案生成 |
| Embedding | BGE (bge-small-zh-v1.5) | 本地向量化，零费用 |
| 向量库 | ChromaDB | 持久化语义检索 |
| 前端 | Streamlit | Web 交互界面 |
| 文档解析 | PyPDF + docx2txt | 多格式文档加载 |

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 安装与运行

```bash
# 1. 克隆项目到本地
git clone https://github.com/<Chris2ai>/multidoc-rag.git
cd multidoc-rag

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置 API Key
# 将 .env.example 复制为 .env，填入你的通义千问 API Key
# 获取地址: https://bailian.console.aliyun.com/

# 6. 运行应用
streamlit run app.py