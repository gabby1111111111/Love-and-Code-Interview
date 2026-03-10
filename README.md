# Love & Code 面试系统 (The Infinite Interview)

**一个为 20k+ AI应用开发岗位量身定制的沉浸式面试演练系统。**

本项目是从企业级多智能体 RAG 系统项目 [Aegis-Isle] 剥离出的独立核心子系统，主要用于展示在 Agent 工作流模型、LangGraph 流程控制、LLM 结构化响应生成以及情景化 RAG 方面的技术储备。它将枯燥的“刷八股文”重塑成了带有“跑团/视觉小说”属性的角色扮演级面试体验。

## 🎯 核心技术与架构能力展示

通过分析这份代码，招聘方可以验证以下核心能力栈：

1. **复杂 Agent Workflow 实现能力 (LangGraph)**
   - 使用 `langgraph` 从零实现状态图 (StateGraph)，实现了完整的双节点循环（`generate_node` 生成问题，`evaluate_node` 判卷打分）。
   - 实现条件路由编排（正确 -> 由 Mentor 进行知识深化扩展，错误 -> 由 Tutor 利用 ELI5 原则拆解讲解）。
2. **LLM 结果结构化控制与双重对话编排 (Polyphonic Prompting)**
   - 自定义封装 `LLMGenerator`，利用结构化 Prompting 将非确定性输出稳定约束为 JSON。
   - 实现主面试官与场外导师的“高并发双声道生成”（`generate_dual_question_interaction`），通过 `asyncio` 异步提高生成吞吐率。
3. **记忆遗忘曲线与 RAG (Knowledge Engine)**
   - 融合艾宾浩斯记忆原理（Box 机制，按 1/3/7/14/30 天周期调度），打造了智能化的 `get_next_question` 推荐算法，综合了失败惩罚、历史答题率等因素。
   - 内置轻量级文件处理流程，利用 LangChain 配合 LLM 对目标 JD （Job Description）和业务知识文档（KB）自动切割、生题、去重、入库。
4. **个性化 Persona 系统设计**
   - 完整适配并接入了开源社区成熟的 `SillyTavern` (V2) 和 PNG 角色隐写卡片格式，具备处理解析超长 World Lore (世界设定薄) 并融合到技术解答中的能力。

## 🌲 与父项目 Aegis-Isle 的血统关联

`Love-and-Code-Interview` 原为 **[Aegis-Isle 系统]** （一个提供数据分析与企业知识管理的 Multi-Agent 协作网络）的辅助子模块 `interview/`：
- **RAG 引擎共用**：在此孤岛项目中保留了 `aegis_isle.rag` 中对于文件加载、Chunking 以及基础生成等核心功能。
- **降级依赖**：原版运行强依赖于 Aegis 中的向量数据库 (Qdrant) 与中枢数据总线。当前版本为了在 GitHub 进行个人能力展示，将所有动态数据流改造为可落盘的无状态 JSON 以实现 `Zero-Config`。

---

## ⚡ 快速启动指南

此项目设计为可一键运行，无需额外搭建数据库：

### 1) 环境要求
* 建议安装 Python 3.10+

### 2) 下载与安装
```bash
git clone https://github.com/gabby1111111111/Love-and-Code-Interview.git
cd Love-and-Code-Interview

# 建议在虚拟环境中运行
python -m venv .venv

# Windows 激活方式:
.venv\Scripts\activate
# Linux/Mac 激活方式:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3) 必要的配置 (.env)
在项目根目录创建一个 `.env` 文件，输入你的大模型密钥（默认兼容 OpenAI/DeepSeek 等标准库）：
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxx

# 可随意指定使用的底层大模型名称
DEFAULT_LLM_MODEL=gpt-4-turbo
```

### 4) 一键启动！
```bash
python scripts/run_interview_app.py
```
> 若出现路径引入异常，本脚本会自动修改 PYTHONPATH 指向 `src/` 以纠正 Python 的环境寻找。

---

## 📂 仓库全貌导航
```text
.
├── data                        -> 数据持久化 (人物卡片 / 题库 / 图像资源)
├── default                     -> 初始化自带的默认配置模版文件
├── frontend
│   └── interview_app.py        -> 基于 Streamlit 撰写的全屏视觉小说 UI 呈现层
├── logs                        -> Loguru 日志切割与异常追溯存档
├── scripts                     -> 维护工具与项目启动脚本
└── src
    └── aegis_isle
        ├── core                -> Base Utils (Logging, Base Configs)
        ├── interview           -> 本次剥离出的核心 (Knowledge, Persona, LangGraph, LLM)
        └── rag                 -> 父工程底层模块 (LLMGen, TextSplitter, Event等依赖支持)
```

---

## 💡 开发规划 / To-Do

1. 接入 **DeepSeek-R1** 专门用于逻辑判决与思考（`evaluate_node` ）。
2. 将 `st.session_state` 持久化迁移到 SQLite，增强长时间会话记忆回溯。
3. 提供 Dockerfile，支持云平台 `Render / HuggingFace Spaces` 秒级部署。

*“技术面试不该只是干瘪的背诵，而应该是一场用代码武装自己的末日求生之旅。”*
