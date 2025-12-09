项目 KT 正式版 Markdown 文档（可直接用）

# 项目核心 KT 文档（Python 技术栈）

本项目采用 Python 作为核心开发语言，围绕 **MCP、AI 管控、可观测、RAG、Agent、UI** 六大模块构建企业级智能体体系结构。本文档用于帮助新同学快速上手整体架构、调用链路与关键组件。

---

## 1. 架构总览

整体链路（自下而上）：


MCP Server → One-API → Langfuse/Langsmith → RAG → Agent(LangChain/LangGraph) → UI(Agent Chat UI)


核心特性：

- **模块化**：每一层可独立替换（如 GraphRAG、不同 LLM、不同工具）
- **企业级可观测性**：Langfuse 全链路追踪
- **可管控**：通过 One-API 实现统一 key 管控、配额控制、模型切换
- **智能工具调用**：MCP server 暴露内部运维能力给 LLM
- **多 Agent 协作**：基于 LangGraph 的 State Machine + Multi-Agent

---

## 2. 核心模块说明

### 2.1 MCP（Model Context Protocol）
- 用于提供公司内部 **运维/业务工具能力** 给智能体调用。
- MCP Hub：https://github.com/samanhappy/mcphub  
  作用：统一注册、调度、暴露所有 MCP 工具。

推荐场景：
- 工单创建
- 查询 CMDB
- 触发部署任务
- 查询 Jira/内部系统

**POC 样例代码（定义 MCP Tool）：**
```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

server = Server("my-mcp-server")

# 定义一个创建工单的工具
@server.call_tool()
async def create_ticket(title: str, description: str, priority: str = "medium") -> TextContent:
    """创建一个TICKET"""
    ticket_data = {
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open"
    }
    # 调用内部 API 或数据库
    result = await save_ticket(ticket_data)
    return TextContent(text=json.dumps(result))

# 定义查询 CMDB 的工具
@server.call_tool()
async def query_cmdb(asset_id: str) -> TextContent:
    """查询资产信息"""
    asset_info = await fetch_from_cmdb(asset_id)
    return TextContent(text=json.dumps(asset_info))

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

---

### 2.2 One-API（AI 管控层）
- 统一管理所有 LLM Key、Quota、QPS、优先级队列
- GitHub：https://github.com/songquanpeng/one-api
- 能作为整个企业的 **AI Gateway**，支持模型：
  - OpenAI / Azure / DeepSeek / Claude / Moonshot 等

作用：
- 降低成本（自动选择最优模型）
- 提供可控性（限流、审计、自动切 provider）

**POC 样例代码（配置 One-API 并调用 LLM）：**
```python
import openai

# 配置 One-API 网关
# One-API 提供一个宝石 OPENAI_API_KEY
openai.api_base = "https://your-one-api-gateway.com/v1"
openai.api_key = "sk-one-api-key-from-dashboard"

# 使用 One-API 调用 LLM
async def call_llm(prompt: str, model: str = "gpt-4"):
    response = await openai.ChatCompletion.acreate(
        model=model,  # One-API 会自动路由到最早可用的提供商
        messages=[
            {"role": "system", "content": "你是一个有效的助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# 控制配额和速率限制
# One-API 提供了 API 控制台：
# 1. 配置 Key 的月配额
# 2. 设置 QPS 限制
# 3. 设置优先级，优先使用低价模型
```

---

### 2.3 可观测（Langsmith + Langfuse）
- 测试环境：Langsmith  
- 线上环境：Langfuse

作用：
- 全链路日志（token、延迟、错误、工具调用）
- 调试 agent、rag pipeline
- 追踪一次用户输入 → agent → tool → llm → 输出 的完整路径

**POC 样例代码（集成 Langfuse 追踪）：**
```python
from langfuse import Langfuse
from langchain.callbacks.manager import CallbackManager
from langchain_core.callbacks import LangfuseCallbackHandler

# 初始化 Langfuse 客户端
langfuse_handler = LangfuseCallbackHandler(
    public_key="your-langfuse-public-key",
    secret_key="your-langfuse-secret-key",
    host="https://cloud.langfuse.com"  # 或你的自托管 URL
)

# 将 Langfuse 处理程序集成到 LangChain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

chat = ChatOpenAI(
    model="gpt-4",
    callbacks=[CallbackManager([langfuse_handler])]
)

# 所有的 LLM 调用、工具执行等都会自动记录到 Langfuse
response = await chat.apredict(input="准备一段技术教教")

# Langfuse 提供了一个控制台
# 你可以查看：
# - 所有 traces 和子 span
# - Token 使用量
# - 错误追踪
# - 性能分析
```

---

### 2.4 RAG 层（LightRAG / GraphRAG）
- 项目当前使用 LightRAG（轻量、快速）
- GitHub：https://github.com/HKUDS/LightRAG
- GraphRAG（LangGraph + 构图式检索）可自行深入

功能：
- 文档→Chunk→Embedding→向量库→查询→过滤→返回给 Agent

**POC 样例代码（使用 LightRAG 实现 RAG 管道）：**
```python
import os
from lightrag import LightRAG
from lightrag.llm import OpenAILLMImpl
from lightrag.embedding import OpenAIEmbeddingImpl

# 初始化 LightRAG
async def initialize_lightrag(work_dir: str = "./rag_storage"):
    """初始化 LightRAG 实例"""
    
    # 配置 LLM（使用 OpenAI 或 One-API）
    llm = OpenAILLMImpl(
        model_name="gpt-4",
        api_key="your-api-key",  # 或使用 One-API 网关
        api_base="https://your-one-api-gateway.com/v1"  # One-API 网关地址
    )
    
    # 配置 Embedding 模型
    embedding = OpenAIEmbeddingImpl(
        model_name="text-embedding-3-small",
        api_key="your-api-key",
        api_base="https://your-one-api-gateway.com/v1"
    )
    
    # 初始化 LightRAG
    rag = LightRAG(
        working_dir=work_dir,
        llm_impl=llm,
        embedding_impl=embedding
    )
    
    return rag

# 添加文档到 RAG
async def add_documents(rag, file_path: str):
    """将文档添加到 LightRAG"""
    
    # 支持多种格式：PDF、TXT、Markdown 等
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 插入文本到 RAG（会自动进行 chunking、embedding、构图）
    await rag.insert(text)
    print(f"Document {file_path} added to RAG")

# 查询 RAG
async def query_rag(rag, question: str, param: dict = None):
    """查询 RAG"""
    
    # LightRAG 支持多种查询模式
    if param is None:
        param = {
            "mode": "local",  # 'local' 或 'global' 模式
            "top_k": 3  # 返回前 k 个结果
        }
    
    # 执行查询
    result = await rag.query(question, param=param)
    
    return result

# 刷新 RAG（更新图和嵌入）
async def refresh_rag(rag):
    """刷新 RAG 的知识图"""
    await rag.refresh()
    print("RAG knowledge graph refreshed")

# 完整流程
async def main():
    # 1. 初始化 LightRAG
    rag = await initialize_lightrag(work_dir="./rag_db")
    
    # 2. 添加文档
    await add_documents(rag, "./company_docs.txt")
    # 可以添加多个文档
    await add_documents(rag, "./business_manual.md")
    
    # 3. 刷新知识图（可选，但推荐在添加多个文档后执行）
    await refresh_rag(rag)
    
    # 4. 查询
    question = "你的公司的核心业务是什么？"
    result = await query_rag(rag, question, param={"mode": "local", "top_k": 5})
    
    print(f"Question: {question}")
    print(f"Answer: {result}")
    
    # 5. 全局查询（可选，性能略差但覆盖面更广）
    global_result = await query_rag(rag, question, param={"mode": "global"})
    print(f"Global Search Result: {global_result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**LightRAG 的优势：**
- **轻量级**：相比 GraphRAG，LightRAG 更轻量、更快
- **自动图构建**：自动从文本提取实体和关系，构建知识图
- **多模式查询**：支持 local（本地）和 global（全局）两种查询模式
- **混合检索**：结合向量检索和图检索，提高准确率
- **易于集成**：可轻松与 LangChain、LangGraph 集成

---

### 2.5 Agent（LangChain / LangGraph）
- 构建智能体
- 支持单 Agent、多 Agent、工具调用
- LangGraph 提供 State Machine 精准控制 Agent 流程

最新特性：
- LangChain 1.0 Middleware（支持日志、审计、上下文压缩）
- Multi-agent 协作：规划模块、推理模块、执行模块拆分

**POC 样例代码（基于 LangGraph 的 Agent）：**
```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import TypedDict, List
import json

# 定义 State
class AgentState(TypedDict):
    user_input: str
    thought: str
    action: str
    observation: str
    final_answer: str
    memory: List[dict]

# 定义一些工具
def create_tools():
    def query_db(question: str) -> str:
        """Query internal database"""
        return json.dumps({"result": "Database result for: " + question})
    
    def call_api(endpoint: str) -> str:
        """Call external API"""
        return json.dumps({"status": "success", "data": "API response"})
    
    tools = [
        Tool(name="QueryDB", func=query_db, description="Query database"),
        Tool(name="CallAPI", func=call_api, description="Call external API")
    ]
    return tools

# 定义 Agent 的逻辑
def think(state: AgentState) -> AgentState:
    """Agent 思考阶段"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"User input: {state['user_input']}\\nThink step by step."
    thought = llm.predict(text=prompt)
    state["thought"] = thought
    return state

def act(state: AgentState) -> AgentState:
    """Agent 执行阶段"""
    tools = create_tools()
    # 根据 thought 选择工具
    tool_map = {tool.name: tool for tool in tools}
    # 这里会根据 LLM 的输出调用相应的工具
    state["action"] = "QueryDB"
    state["observation"] = tool_map[state["action"]]({\"question\": state[\"user_input\"]})
    return state

def conclude(state: AgentState) -> AgentState:
    """Agent 总结阶段"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"Based on observation: {state['observation']}, provide final answer."
    state["final_answer"] = llm.predict(text=prompt)
    return state

# 构建 State Machine
def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("conclude", conclude)
    
    graph.add_edge("think", "act")
    graph.add_edge("act", "conclude")
    graph.add_edge("conclude", END)
    
    graph.set_entry_point("think")
    return graph.compile()

# 使用
if __name__ == "__main__":
    agent = build_agent_graph()
    result = agent.invoke({"user_input": "查询最新的项目进展"})
    print(result["final_answer"])
```

---

### 2.6 UI（Agent Chat UI / 内部 AG-UI）
- 临时：Agent Chat UI
- 公司内部已有 AG-UI 封装（后续统一改造）

作用：
- 多 Agent 对话展示
- 工具流可视化
- token 测量、延迟分析

**POC 样例代码（简单的 Flask + WebSocket UI）：**
```python
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import asyncio
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 应界信息
class Message:
    def __init__(self, role: str, content: str, timestamp: str = None, tokens: int = 0):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.tokens = tokens

# 存储每个用户的对话历史
conversations = {}

@app.route("/")
def index():
    return render_template("chat.html")

@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")
    conversations[request.sid] = []

@socketio.on("send_message")
async def handle_message(data):
    user_id = request.sid
    user_message = data.get("message")
    agent_name = data.get("agent", "default")
    
    # 初始化每个客户端的对话
    if user_id not in conversations:
        conversations[user_id] = []
    
    # 保存用户消息
    user_msg = Message(role="user", content=user_message)
    conversations[user_id].append(user_msg)
    
    # 发送用户消息到前端
    emit("message", {
        "role": "user",
        "content": user_message,
        "timestamp": user_msg.timestamp
    }, to=user_id)
    
    # 调用 Agent
    try:
        # 这里可以接入你的 Agent会话
        agent_response = await call_agent(agent_name, user_message)
        
        # 保存 Agent 响应
        agent_msg = Message(
            role="assistant",
            content=agent_response,
            tokens=estimate_tokens(agent_response)
        )
        conversations[user_id].append(agent_msg)
        
        # 发送 Agent 响应到前端
        emit("message", {
            "role": "assistant",
            "content": agent_response,
            "timestamp": agent_msg.timestamp,
            "tokens": agent_msg.tokens
        }, to=user_id)
    except Exception as e:
        emit("error", {"message": str(e)}, to=user_id)

async def call_agent(agent_name: str, message: str):
    """Call the Agent with message"""
    # 接入你的 Agent 招聘
    # 可以是 LangGraph Agent 或 LangChain Agent
    return f"Agent '{agent_name}' response to: {message}"

def estimate_tokens(text: str) -> int:
    """Estimate token count"""
    return len(text.split()) // 4  # 简单估算

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)

# HTML 模板 (templates/chat.html)
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Agent Chat UI</title>
#     <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
# </head>
# <body>
#     <div id="chat-container">
#         <div id="messages"></div>
#         <input type="text" id="message-input" placeholder="你想说什么？">
#         <button onclick="sendMessage()">Send</button>
#     </div>
#     <script>
#         const socket = io();
#         
#         socket.on('message', (data) => {
#             const msgDiv = document.createElement('div');
#             msgDiv.className = 'message ' + data.role;
#             msgDiv.innerHTML = `
#                 <strong>${data.role}</strong>: ${data.content}
#                 ${data.tokens ? `<span class="tokens">${data.tokens} tokens</span>` : ''}
#             `;
#             document.getElementById('messages').appendChild(msgDiv);
#         });
#         
#         function sendMessage() {
#             const input = document.getElementById('message-input');
#             socket.emit('send_message', {message: input.value});
#             input.value = '';
#         }
#     </script>
# </body>
# </html>
```

---

## 3. 端到端调用链路（数据流）

```
User Input
    ↓
    UI (AG-UI / Flask WebSocket)
    ↓
    Agent (LangGraph State Machine)
    ↓
    Thought → Action → Observation → Final Answer
    ↓
    Tool Calls (MCP Server)
    → Query DB / API / etc.
    ↓
    RAG Pipeline (when needed)
    → Document Retrieval → Context Injection
    ↓
    LLM Call (via One-API)
    → Provider Routing (DeepSeek/OpenAI/Claude)
    ↓
    Langfuse Tracing
    → Log Token Usage, Latency, Tool Calls
    ↓
    Agent Response Aggregation
    ↓
    Final Response → UI Display
```

**完整 POC 示例（整合调用）：**
```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain_core.callbacks import LangfuseCallbackHandler

# 初始化 Langfuse
langfuse_handler = LangfuseCallbackHandler(
    public_key="your-key",
    secret_key="your-secret"
)

class FullStackAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_base="https://your-one-api-gateway.com/v1",
            api_key="your-one-api-key",
            callbacks=[CallbackManager([langfuse_handler])]
        )
        self.rag_chain = None  # 来自 RAG 模块
        self.tools = {}  # 来自 MCP Server
    
    async def process_user_input(self, user_input: str):
        print(f"[User]: {user_input}")
        
        # 步骤 1: Agent 思考
        thought = await self.llm.apredict(
            f"Think about how to answer: {user_input}"
        )
        print(f"[Agent Thought]: {thought}")
        
        # 步骤 2: 判断是否需要查询 RAG
        if "query" in user_input.lower():
            context = await self.rag_chain.arun(user_input)
            print(f"[RAG Context]: {context}")
        else:
            context = ""
        
        # 步骤 3: Agent 选择工具或直接回答
        # 来自会话历史 + Langfuse 挂钩 = 全链路可观测
        
        # 步骤 4: 生成最终答案
        final_answer = await self.llm.apredict(
            f"Based on thought: {thought} and context: {context}, answer: {user_input}"
        )
        print(f"[Final Answer]: {final_answer}")
        
        return final_answer

# 使用
if __name__ == "__main__":
    agent = FullStackAgent()
    answer = await agent.process_user_input("查询最新的业务数据")
```
