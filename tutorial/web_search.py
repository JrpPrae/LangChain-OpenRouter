# code นี้รันได้ ไม่ error

# -------------------------
# Web Search (Built-in Tools)
#     is tools of OpenAI to help AI model able to real-time 
#     search information on internet (Search Engine) ซึ่งจะช่วยให้ AI ตองคำถามหรือ
#     ให้ข้อมูลอัปเดตล่าสุดหรือข้อมูลเฉพาะทางได้
#     ความสามารถ :
#     - ค้นหาข้อมูลล่าสุด
#     - ค้นหาข้อมูลเฉพาะทาง
#     - ตรวจสอบข้อมูลได้
# --------------------------

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ---------------------------------------
# 1. Create llm model through OpenRouter
# ---------------------------------------
llm = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    temperature=0.7, 
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

'''tools
tools = {"type":"web_search_preview"} # ใช้ไม่ได้
ติดเครื่องมือใน AI
llm_search = llm.bind_tools([tools])
calling model
response = llm_search.invoke("What is the llm")
print(response.content)'''

# ------------------------
# tools = {"type":"web_search_preview"} นี้ สามารถใช้กับ openAI function calling โดยตรงได้ 
# แต่ OpenRouter ไม่รองรับการ bind tools ด้วยการใช้ bind_tools()
# วิธีแก้คือ ต้องสร้าง wrapper เอง
# -------------------------

# ----------------------------
# 2. Tool (ต้องมี docstring)
# tavily คือ api websearch
# ----------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for real-time information using Tavily API and return results."""
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "num_results": 3
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        results = [item["content"] for item in data.get("results", [])]
        return "\n".join(results) if results else "(No results found)"
    else:
        return f"Error from Tavily API: {response.text}"

# ----------------------------
# 3. Bind tool
# ----------------------------
llm_with_tools = llm.bind_tools([web_search])

# ----------------------------
# 4. เรียกใช้งาน
# ----------------------------
query = "นายกของประเทศไทยคือใคร"
response = llm_with_tools.invoke(query)

# LangChain จะคืน AIMessage ไม่ใช่ str
print("🤖 LLM Output:\n", response)
print("🤖 LLM Content:\n", response.content)

