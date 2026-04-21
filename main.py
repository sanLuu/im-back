import json
import re
import os
import pickle
import requests
import faiss
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
import shlex

# ====================== 关键：添加 Hugging Face 国内镜像 ======================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# ==========================================================================

from sentence_transformers import SentenceTransformer

# ====================== 配置区 ======================
API_KEY = "sk-d27d6a9ff980479e81a517ac6188f23e"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-turbo"
EMBED_MODEL = "all-MiniLM-L6-v2"
MEMORY_DIR = "./agent_memory"
FAISS_INDEX_FILE = os.path.join(MEMORY_DIR, "faiss.index")
MEMORY_DATA_FILE = os.path.join(MEMORY_DIR, "memory_data.pkl")
# ====================================================

os.makedirs(MEMORY_DIR, exist_ok=True)

# ====================== 向量记忆懒加载 ======================
vector_memory = None

def get_vector_memory():
    global vector_memory
    if vector_memory is None:
        print(f"{Color.WARNING}[💾 正在加载向量模型与记忆库...] {Color.ENDC}")
        vector_memory = VectorMemory(EMBED_MODEL)
    return vector_memory

def preload_model():
    get_vector_memory()

# ====================== 终端颜色（兼容Windows） ======================
class Color:
    if os.name == "nt" and not os.environ.get("ANSICON") and not os.environ.get("WT_SESSION"):
        HEADER = ''; OKBLUE = ''; OKCYAN = ''; OKGREEN = ''; WARNING = ''; FAIL = ''; ENDC = ''; BOLD = ''; UNDERLINE = ''
    else:
        HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
        WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

@dataclass
class Tool:
    name: str
    func: callable
    desc: str
    params: List[str]

# ====================== 工具定义（本地知识库搜索+计算+时间） ======================
def search_web(query: str) -> str:
    print(f"{Color.OKBLUE}[🔍 工具调用] search_web: {query}{Color.ENDC}")
    knowledge_base = {
        "AI Agent": "AI Agent（人工智能智能体）是一种能够自主感知环境、做出决策并执行行动的AI系统，核心是ReAct框架（思考-行动-观察），主流框架包括LangChain、LlamaIndex、AutoGen、MetaGPT，部署常用FastAPI+Docker+FAISS做向量记忆。",
        "通义千问": "通义千问是阿里云开发的大语言模型，支持API调用，兼容OpenAI接口格式，可用于构建AI Agent、对话系统等应用。",
        "ReAct": "ReAct是AI Agent的经典推理框架，核心是「思考(Reason)→行动(Act)→观察(Observe)」循环，让大模型结合工具调用完成复杂任务。",
        "LangChain": "LangChain是最流行的AI Agent开发框架，提供LLM封装、工具调用、记忆管理、链式调用等核心能力，支持快速构建生产级Agent。"
    }
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key.lower() in query_lower:
            return value
    return f"未在本地知识库中找到关于「{query}」的相关信息，可手动补充知识库内容。"

def calculate(expr: str) -> str:
    print(f"{Color.OKBLUE}[🧮 工具调用] calculate: {expr}{Color.ENDC}")
    try:
        expr = expr.strip()
        if not expr:
            return "计算错误：表达式为空"
        expr = re.sub(r"(\d+)的(\d+)次方", r"\1**\2", expr)
        expr = re.sub(r"(\d+)\^(\d+)", r"\1**\2", expr)
        expr = expr.replace("加", "+").replace("减", "-").replace("乘", "*").replace("乘以", "*")
        expr = expr.replace("除", "/").replace("除以", "/")
        chinese_num_map = {"零":0, "一":1, "二":2, "两":2, "三":3, "四":4, "五":5, 
                          "六":6, "七":7, "八":8, "九":9, "十":10}
        for cn_num, arabic_num in chinese_num_map.items():
            expr = expr.replace(cn_num, str(arabic_num))
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expr):
            invalid_chars = [c for c in expr if c not in allowed_chars]
            return f"计算错误：表达式包含非法字符 {set(invalid_chars)}"
        res = eval(expr, {"__builtins__": None}, {})
        return f"{expr} = {res}"
    except Exception as e:
        return f"计算错误：{str(e)}"

def get_time() -> str:
    print(f"{Color.OKBLUE}[⏰ 工具调用] get_time{Color.ENDC}")
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %A %H:%M:%S")

TOOLS: Dict[str, Tool] = {
    "search_web": Tool("search_web", search_web, "联网搜索，参数：query 字符串", ["query"]),
    "calculate": Tool("calculate", calculate, "数学计算，支持中文表达式", ["expression"]),
    "get_time": Tool("get_time", get_time, "获取当前日期时间，无参数", []),
}

TOOL_PROMPT = """
可用工具：
1. search_web(query) - 联网搜索，参数为搜索关键词字符串
2. calculate(expression) - 数学计算，参数为表达式字符串
3. get_time() - 获取当前日期时间，无参数

输出格式严格遵循：
Thought: 你的思考过程
Action: 工具名(参数) 或 Action: 无
"""

# ====================== 向量记忆管理 ======================
class VectorMemory:
    def __init__(self, embed_model_name: str):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.dim = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.memory_data: List[Dict[str, Any]] = []
        self.load_from_disk()

    def load_from_disk(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MEMORY_DATA_FILE):
            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                with open(MEMORY_DATA_FILE, "rb") as f:
                    self.memory_data = pickle.load(f)
                print(f"{Color.OKGREEN}[💾 记忆加载] 成功加载 {len(self.memory_data)} 条历史记忆{Color.ENDC}")
            except Exception as e:
                print(f"{Color.FAIL}[❌ 记忆加载失败] {e}，新建空库{Color.ENDC}")
                self.index = faiss.IndexFlatL2(self.dim)
                self.memory_data = []
        else:
            print(f"{Color.WARNING}[💾 记忆加载] 无历史记忆，新建空库{Color.ENDC}")

    def save_to_disk(self):
        try:
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            with open(MEMORY_DATA_FILE, "wb") as f:
                pickle.dump(self.memory_data, f)
        except Exception as e:
            print(f"{Color.FAIL}[❌ 记忆保存失败] {e}{Color.ENDC}")

    def add(self, text: str, role: str = "user"):
        if not text.strip():
            return
        embedding = self.embed_model.encode([text])
        self.index.add(embedding)
        self.memory_data.append({"role": role, "content": text, "timestamp": os.path.getctime(MEMORY_DIR)})
        self.save_to_disk()
        print(f"{Color.OKGREEN}[💾 记忆存储] 已存入{role}内容: {text[:50]}...{Color.ENDC}")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.index.ntotal == 0:
            return []
        query_emb = self.embed_model.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        return [self.memory_data[i]["content"] for i in indices[0] if i < len(self.memory_data)]

    def clear_memory(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.memory_data = []
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
        if os.path.exists(MEMORY_DATA_FILE):
            os.remove(MEMORY_DATA_FILE)
        print(f"{Color.OKGREEN}[🧹 已清空所有本地记忆]{Color.ENDC}")

# ====================== 短期对话记忆 ======================
class ShortMemory:
    def __init__(self, max_history: int = 20):
        self.history = []
        self.max_history = max_history

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if role in ["user", "assistant"]:
            get_vector_memory().add(content, role)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_messages(self, max_round: int = 5) -> List[Dict[str, str]]:
        return self.history[-max_round*2:]

    def clear(self):
        self.history = []

short_memory = ShortMemory()

# ====================== LLM 调用 ======================
def llm_completion(messages: list, retry: int = 2) -> Optional[str]:
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": messages, "temperature": 0.0, "max_tokens": 1000}
    
    for i in range(retry + 1):
        try:
            print(f"{Color.OKCYAN}[🤖 调用 LLM] 正在请求模型... (第{i+1}次尝试){Color.ENDC}")
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            print(f"{Color.OKGREEN}[🤖 LLM 响应] 请求成功{Color.ENDC}")
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"{Color.FAIL}[❌ LLM 调用失败 (第{i+1}次)] {e}{Color.ENDC}")
            if i == retry:
                return None

# ====================== ReAct 解析（强化容错） ======================
def parse_react(text: str) -> Tuple[str, str]:
    thought = ""
    action = ""
    action_match = re.search(r"Action\s*:\s*(.+?)(?=\n\s*Thought\s*:|$)", text, re.DOTALL)
    if action_match:
        action = re.sub(r"\s+", " ", action_match.group(1)).strip()
    thought_match = re.search(r"Thought\s*:\s*(.+?)(?=\n\s*Action\s*:|$)", text, re.DOTALL)
    if thought_match:
        thought = re.sub(r"\s+", " ", thought_match.group(1)).strip()
    if not thought and action:
        thought = text.split("Action:")[0].strip()
    if action and "(" not in action and action != "无":
        if action in TOOLS:
            tool = TOOLS[action]
            if len(tool.params) == 0:
                action = f"{action}()"
            else:
                action = f"{action}('')"
    return thought, action

# ====================== 工具执行（优化错误处理） ======================
def execute_action(action_str: str) -> str:
    try:
        if not action_str or action_str == "无":
            return "无需调用工具"
        if "(" not in action_str or ")" not in action_str:
            return f"动作格式错误，正确格式：工具名(参数)，例如 search_web('AI Agent是什么')"
        tool_name_part, param_part = action_str.split("(", 1)
        tool_name = tool_name_part.strip()
        param_part = param_part.rsplit(")", 1)[0].strip()
        if tool_name not in TOOLS:
            return f"未知工具：{tool_name}，可用工具：{', '.join(TOOLS.keys())}"
        tool = TOOLS[tool_name]
        if len(tool.params) == 0:
            if param_part:
                return f"工具{tool_name}无需参数，请勿传入参数"
            return tool.func()
        else:
            if not param_part:
                return f"参数缺失，{tool_name}需要{len(tool.params)}个参数"
            try:
                params = shlex.split(param_part)
                if len(params) != len(tool.params):
                    return f"参数数量错误，{tool_name}需要{len(tool.params)}个参数，实际传入{len(params)}个"
                return tool.func(*params)
            except Exception as e:
                return f"参数解析错误：{str(e)}"
    except Exception as e:
        return f"执行失败：{str(e)}"

# ====================== Agent 核心逻辑 ======================
def agent_run(query: str, max_round: int = 3) -> str:
    short_memory.add("user", query)
    related_memories = get_vector_memory().retrieve(query)
    memory_context = "\n".join([f"- {mem}" for mem in related_memories]) if related_memories else "无"
    tool_observations = []

    for r in range(max_round):
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in short_memory.get_messages()])
        tool_context = "\n".join([f"工具调用结果：{obs}" for obs in tool_observations]) if tool_observations else "无"

        messages = [
            {"role": "system", "content":
                "你是一个严格遵循 ReAct 框架的 AI Agent，必须100%遵守以下规则：\n"
                "1. 必须严格按照「Thought: ... Action: ...」格式输出，禁止任何多余内容\n"
                "2. 遇到数学计算必须调用calculate工具，遇到未知知识必须调用search_web工具，遇到日期问题必须调用get_time工具\n"
                "3. 工具调用必须严格遵循格式：工具名(参数)，例如 search_web('AI Agent是什么')、calculate('123+456*2')\n"
                "4. 只有当工具返回结果后，才能输出 Action: 无，整理最终答案\n"
                "5. 绝对不要编造答案，必须通过工具获取真实结果\n"
                "6. 最终回答必须清晰、直接，包含工具返回的核心信息"},
            {"role": "user", "content": f"工具说明：\n{TOOL_PROMPT}\n相关历史记忆：{memory_context}\n历史对话：{history_context}\n工具调用结果：{tool_context}\n用户问题：{query}"}
        ]

        llm_out = llm_completion(messages)
        if not llm_out:
            raise HTTPException(status_code=500, detail="LLM 调用失败，请稍后重试")
        
        thought, action = parse_react(llm_out)
        print(f"{Color.OKCYAN}[🤖 Agent思考] {thought}{Color.ENDC}")
        print(f"{Color.OKCYAN}[🤖 Agent动作] {action}{Color.ENDC}")

        if action == "无":
            if tool_observations:
                final_answer = tool_observations[-1]
            elif related_memories:
                final_answer = related_memories[0]
            else:
                print(f"{Color.WARNING}[⚠️ LLM违规] 未调用工具直接返回，继续下一轮{Color.ENDC}")
                continue
            short_memory.add("assistant", final_answer)
            return final_answer
        else:
            obs = execute_action(action)
            if "格式错误" not in obs:
                tool_observations.append(obs)
            short_memory.add("system", f"工具{action}执行结果：{obs}")

    final_answer = "\n".join(tool_observations) if tool_observations else "已达到最大思考轮次，未获取有效结果"
    short_memory.add("assistant", final_answer)
    return final_answer

# ====================== FastAPI 配置 ======================
class ChatRequest(BaseModel):
    query: str
    max_round: int = 3

class ChatResponse(BaseModel):
    query: str
    answer: str

app = FastAPI(title="ReAct Agent API", version="1.0", docs_url=None, redoc=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.staticfile.org/swagger-ui/5.11.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.staticfile.org/swagger-ui/5.11.0/swagger-ui.css",
    )

@app.post("/chat", response_model=ChatResponse, summary="和 ReAct Agent 对话")
async def chat(request: ChatRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        answer = agent_run(request.query, request.max_round)
        return {"query": request.query, "answer": answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")

@app.post("/clear_memory", summary="清空所有记忆")
async def clear_memory():
    get_vector_memory().clear_memory()
    short_memory.clear()
    return {"status": "success", "message": "所有记忆已清空"}

@app.get("/", summary="健康检查")
async def root():
    return {"message": "ReAct Agent API is running!", "status": "healthy"}

# ====================== 启动服务 ======================
if __name__ == "__main__":
    import uvicorn
    preload_model()
    port = 8000
    max_retry = 5
    for i in range(max_retry):
        try:
            print(f"{Color.OKGREEN}[🚀 启动服务] 尝试在端口 {port} 启动服务...{Color.ENDC}")
            uvicorn.run(app, host="127.0.0.1", port=port)
            break
        except Exception as e:
            if "address already in use" in str(e).lower():
                print(f"{Color.WARNING}[⚠️ 端口 {port} 被占用，尝试端口 {port+1}...{Color.ENDC}")
                port += 1
            else:
                print(f"{Color.FAIL}[❌ 服务启动失败] {e}{Color.ENDC}")
                break