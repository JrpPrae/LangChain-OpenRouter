# LangChain × OpenRouter

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Gateway-6566F1)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0467DF?logo=meta&logoColor=white)

รวมตัวอย่างและแบบฝึกหัดการใช้ **LangChain** ร่วมกับ **OpenRouter** ซึ่งเป็น API gateway ที่เรียก LLM ได้หลายค่าย (OpenAI, Anthropic, Google, Meta ฯลฯ) ผ่าน endpoint เดียวที่เข้ากันได้กับ OpenAI API ครอบคลุมตั้งแต่พื้นฐานการเรียกโมเดล, Prompt Template, Streaming, **RAG (Retrieval-Augmented Generation)**, **Tool Calling / Web Search** ไปจนถึง **Search Agent** พร้อมหน้าเว็บด้วย **Streamlit**

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [ทำไมต้อง OpenRouter](#ทำไมต้อง-openrouter)
- [Workflow](#workflow)
  - [1. LLM Chain พื้นฐาน](#1-llm-chain-พื้นฐาน)
  - [2. RAG Pipeline](#2-rag-pipeline)
  - [3. Tool Calling / Web Search](#3-tool-calling--web-search)
  - [4. Search Agent (ReAct)](#4-search-agent-react)
- [รายละเอียดแต่ละไฟล์](#รายละเอียดแต่ละไฟล์)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [วิธีใช้งาน](#วิธีใช้งาน)
- [หมายเหตุและสิ่งที่ควรพัฒนาต่อ](#หมายเหตุและสิ่งที่ควรพัฒนาต่อ)

## ภาพรวม

โปรเจกต์แบ่งเป็น 3 ส่วน:

| ส่วน | เนื้อหา | รูปแบบ |
| --- | --- | --- |
| [`tutorial/`](tutorial/) | บทเรียนแนวคิดหลักของ LangChain: Chat Model, Messages, Streaming, Prompt Template, RAG และ Tool Calling | Jupyter Notebook + Python script |
| [`practice/`](practice/) | แบบฝึกหัดสร้างเว็บแอปเล็ก ๆ ด้วย Streamlit: ถาม-ตอบ, สร้าง Blog Outline, Search Agent | Streamlit app |
| [`example_code/`](example_code/) | ข้อมูลตัวอย่าง (`data.txt`) ของบริษัทสมมติ "ABC" สำหรับทดลอง RAG | Text file |

ทุกตัวอย่างใช้ `ChatOpenAI` จาก `langchain-openai` แต่ชี้ `base_url` ไปที่ OpenRouter:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("MODEL"),                  # เช่น "openai/gpt-4o"
    temperature=0.7,
    max_tokens=500,
    base_url="https://openrouter.ai/api/v1",   # เปลี่ยนจาก OpenAI เป็น OpenRouter
)
```

## ทำไมต้อง OpenRouter

- **API เดียว หลายโมเดล** — เปลี่ยนโมเดลได้โดยแก้แค่ชื่อโมเดล (`openai/gpt-4o`, `anthropic/...`, `google/...`) ไม่ต้องแก้โค้ด
- **เข้ากันได้กับ OpenAI API** — ใช้ `ChatOpenAI` ของ LangChain ได้ทันที เพียงเปลี่ยน `base_url`
- **ข้อจำกัดที่พบในโปรเจกต์นี้** — OpenRouter **ไม่มี Embeddings API** (`OpenAIEmbeddings` ใช้ไม่ได้) และไม่รองรับ built-in tool ของ OpenAI อย่าง `web_search_preview` จึงต้องใช้ทางเลือกอื่นแทน (รายละเอียดในหัวข้อ [RAG](#2-rag-pipeline) และ [Tool Calling](#3-tool-calling--web-search))

## Workflow

### 1. LLM Chain พื้นฐาน

ไฟล์: [`tutorial/llm_chain.ipynb`](tutorial/llm_chain.ipynb)

```text
โหลด .env (MODEL, OPENAI_API_KEY)
        │
        ▼
สร้าง ChatOpenAI(base_url = OpenRouter)
        │
        ├─▶ model.invoke([SystemMessage, HumanMessage])   → AIMessage (.content = "สวัสดี")
        │
        ├─▶ model.stream(messages)                        → ทยอยรับทีละ token: ส|วั|สด|ี
        │
        └─▶ ChatPromptTemplate
               ("system", "Translate the following from Thai into {language}")
               ("user",   "{text}")
                    │ .invoke({"language": "Japan", "text": "สวัสดี"})
                    ▼
               ChatPromptValue → model.invoke() → "こんにちは"
```

### 2. RAG Pipeline

ไฟล์: [`tutorial/rag.ipynb`](tutorial/rag.ipynb) — ตอบคำถามโดยอ้างอิงข้อมูลจากเอกสาร [`example_code/data.txt`](example_code/data.txt)

```text
┌──────────────────────────── Indexing (ทำครั้งเดียว) ────────────────────────────┐
│                                                                                │
│  data.txt ──▶ TextLoader ──▶ RecursiveCharacterTextSplitter ──▶ 9 chunks        │
│  (ข้อมูลบริษัท ABC)            chunk_size=100, chunk_overlap=20                   │
│                                        │                                       │
│                                        ▼                                       │
│              HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)     │
│                                        │  แปลงแต่ละ chunk เป็น vector (รันบนเครื่อง) │
│                                        ▼                                       │
│                              FAISS Vector Store                                │
└────────────────────────────────────────┬───────────────────────────────────────┘
                                         │ .as_retriever()
┌──────────────────────────── Query (ทุกครั้งที่ถาม) ─────────────────────────────┐
│                                        ▼                                       │
│  คำถาม ──┬──▶ Retriever ─── ค้น chunk ที่ใกล้เคียงที่สุด ──▶ {context}            │
│          └──▶ RunnablePassthrough ──────────────────────▶ {question}           │
│                                        │                                       │
│                                        ▼                                       │
│  ChatPromptTemplate                                                            │
│    system: "ใช้ข้อมูลจากเอกสารในการตอบคำถาม ... สั้นกระชับ สุภาพเป็นกันเอง"          │
│    human : "คำถาม: {question}, ข้อมูลที่เกี่ยวข้อง: {context}"                     │
│                                        │                                       │
│                                        ▼                                       │
│  ChatOpenAI (OpenRouter) ──▶ StrOutputParser ──▶ คำตอบ (string)                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

Chain เขียนด้วย **LCEL (LangChain Expression Language)** ใช้ตัวดำเนินการ `|` ต่อขั้นตอนเข้าด้วยกัน:

```python
rag_chain = (
    {"context": retrievers, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("มีสินค้าอะไรบ้าง")
# 'จากข้อมูลที่ให้มา สินค้าที่มีได้แก่ หนังสือการ์ตูนประเภทต่างๆ เช่น มังงะ คอมิกส์ ไลท์โนเวล และนิยายภาพค่ะ ...'
```

| ขั้นตอน | Component | หน้าที่ |
| --- | --- | --- |
| Document Loader | `TextLoader` | โหลดไฟล์ข้อความ (UTF-8) เป็น `Document` |
| Text Splitter | `RecursiveCharacterTextSplitter` | แบ่งตามโครงสร้างข้อความ (ย่อหน้า → บรรทัด → คำ) chunk ละไม่เกิน 100 ตัวอักษร ซ้อนทับ 20 ตัวอักษรเพื่อรักษาบริบท |
| Embeddings | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) | แปลง chunk เป็น vector 384 มิติ — ใช้แทน `OpenAIEmbeddings` ที่ OpenRouter ไม่รองรับ |
| Vector Store | `FAISS` | เก็บ vector ใน memory และค้นหาด้วย similarity search |
| Retriever | `vectorstore.as_retriever()` | รับคำถาม → คืน chunk ที่เกี่ยวข้องที่สุด |
| Chain | LCEL + `StrOutputParser` | ประกอบ prompt → LLM → แปลงผลเป็น string |

### 3. Tool Calling / Web Search

ไฟล์: [`tutorial/web_search.py`](tutorial/web_search.py)

```text
@tool web_search(query)  ── เรียก Tavily Search API (https://api.tavily.com/search)
        │                    คืนเนื้อหาผลค้นหาต่อกันเป็นข้อความ
        ▼
llm.bind_tools([web_search])   ── แนบ schema ของ tool ให้โมเดลรู้จัก
        │
        ▼
llm_with_tools.invoke("นายกของประเทศไทยคือใคร")
        │
        ▼
AIMessage  ── โมเดลตัดสินใจว่าจะเรียก tool หรือไม่ (ดูได้ที่ response.tool_calls)
```

เนื่องจาก OpenRouter ไม่รองรับ built-in tool `{"type": "web_search_preview"}` ของ OpenAI จึงสร้าง **custom tool** ด้วย decorator `@tool` (ต้องมี docstring เพื่อใช้เป็นคำอธิบาย tool) แล้วเรียก Tavily API แทน

### 4. Search Agent (ReAct)

ไฟล์: [`practice/3_Langchain_search.py`](practice/3_Langchain_search.py)

```text
ผู้ใช้ใส่ Serper API Key (sidebar) + พิมพ์คำถาม
        │
        ▼
ตรวจสอบ key ครบหรือไม่ → ถ้าไม่ครบแสดง st.info แจ้งเตือน
        │
        ▼
Agent (ZERO_SHOT_REACT_DESCRIPTION)
   ┌──▶ Thought  : โมเดลคิดว่าต้องทำอะไร
   │    Action   : เรียก tool "Intermediate Answer" (GoogleSerperAPIWrapper → Google Search)
   │    Observation: ผลการค้นหา
   └─── วนซ้ำจนได้คำตอบ
        │
        ▼
Final Answer → แสดงใน Streamlit ("### Answer")
```

## รายละเอียดแต่ละไฟล์

| ไฟล์ | สิ่งที่เรียนรู้ / ทำ | วิธีรัน |
| --- | --- | --- |
| [`tutorial/llm_chain.ipynb`](tutorial/llm_chain.ipynb) | `ChatOpenAI` + OpenRouter, `SystemMessage`/`HumanMessage`, `.invoke()`, `.stream()`, `ChatPromptTemplate` (แปลภาษา ไทย→ญี่ปุ่น) | Jupyter |
| [`tutorial/rag.ipynb`](tutorial/rag.ipynb) | ทฤษฎีและโค้ด RAG ครบ 6 ขั้น: Loader → Splitter → Embeddings → FAISS → Retriever → Chain | Jupyter |
| [`tutorial/web_search.py`](tutorial/web_search.py) | สร้าง custom tool ด้วย `@tool`, เรียก Tavily API, `bind_tools()` | `python tutorial/web_search.py` |
| [`practice/1_Langchain_quickstart.py`](practice/1_Langchain_quickstart.py) | แอปถาม-ตอบพื้นฐาน: `st.form` + `st.text_area` → `ChatOpenAI.invoke()` | `streamlit run` |
| [`practice/2_Langchain_PromptTemplate.py`](practice/2_Langchain_PromptTemplate.py) | สร้าง Blog Outline จากหัวข้อ ด้วย `ChatPromptTemplate` + LCEL (`prompt \| llm`) | `streamlit run` |
| [`practice/3_Langchain_search.py`](practice/3_Langchain_search.py) | Search Agent ด้วย Google Serper + ReAct agent | `streamlit run` |
| [`example_code/data.txt`](example_code/data.txt) | ข้อมูลบริษัทสมมติ ABC (สินค้า, กลุ่มเป้าหมาย, ช่องทางขาย, ข้อมูลติดต่อ) | — |

## Tech Stack

| หมวด | เทคโนโลยี | หน้าที่ในโปรเจกต์ |
| --- | --- | --- |
| ภาษา | **Python 3.12** | ภาษาหลัก |
| LLM Framework | **LangChain 0.3** (`langchain`, `langchain-core`, `langchain-community`) | Prompt Template, LCEL Chain, RAG components, Agent, Tool |
| LLM Client | **langchain-openai** (`ChatOpenAI`) | เรียก Chat Completions API (ชี้ไปที่ OpenRouter) |
| LLM Gateway | **OpenRouter** | ให้บริการ LLM หลายค่ายผ่าน API เดียว (ในตัวอย่างใช้ `openai/gpt-4o`, `openai/gpt-4o-mini`) |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` ผ่าน `HuggingFaceEmbeddings` | แปลงข้อความเป็น vector บนเครื่อง ไม่ต้องใช้ API |
| Vector Store | **FAISS** (`faiss-cpu` / `faiss-gpu`) | เก็บและค้นหา vector แบบ similarity search |
| Web Search | **Tavily API**, **Google Serper API** | ค้นหาข้อมูล real-time จากอินเทอร์เน็ต |
| Web UI | **Streamlit** | หน้าเว็บของแบบฝึกหัดใน `practice/` |
| Notebook | **Jupyter** | รัน `tutorial/*.ipynb` |
| Configuration | **python-dotenv** | โหลด API key และชื่อโมเดลจาก `.env` |
| HTTP Client | **requests** | เรียก Tavily API ใน `web_search.py` |

## Environment Variables

สร้างไฟล์ `.env` ที่โฟลเดอร์ที่ใช้รันคำสั่ง (`load_dotenv()` จะค้นหาไฟล์ `.env` จากโฟลเดอร์ของสคริปต์ขึ้นไปยังโฟลเดอร์แม่):

```env
# OpenRouter — ChatOpenAI อ่านตัวแปรนี้อัตโนมัติ ใส่ key ของ OpenRouter (sk-or-...)
OPENAI_API_KEY=sk-or-xxxxxxxxxxxxxxxx
MODEL=openai/gpt-4o

# ใช้ใน tutorial/web_search.py
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

| ตัวแปร | ใช้ใน | คำอธิบาย |
| --- | --- | --- |
| `OPENAI_API_KEY` | notebooks, `practice/*` | **API key ของ OpenRouter** (ตั้งชื่อตามที่ `ChatOpenAI` อ่านจาก environment โดยอัตโนมัติ) — สร้างได้ที่ [openrouter.ai/keys](https://openrouter.ai/keys) |
| `MODEL` | notebooks, `practice/*` | ชื่อโมเดลตามรูปแบบของ OpenRouter เช่น `openai/gpt-4o` — ดูรายชื่อได้ที่ [openrouter.ai/models](https://openrouter.ai/models) |
| `OPENROUTER_API_KEY` | `tutorial/web_search.py` | API key ของ OpenRouter (ส่งผ่านพารามิเตอร์ `api_key` โดยตรง) |
| `TAVILY_API_KEY` | `tutorial/web_search.py` | API key ของ [Tavily](https://tavily.com/) |
| Serper API Key | `practice/3_Langchain_search.py` | **ไม่ได้อ่านจาก `.env`** — กรอกในช่อง sidebar ของแอป สมัครได้ที่ [serper.dev](https://serper.dev/) |

## โครงสร้างโปรเจกต์

```text
LangChain-OpenRouter/
├── tutorial/
│   ├── llm_chain.ipynb                  # Chat model, messages, streaming, prompt template
│   ├── rag.ipynb                        # RAG pipeline: TextLoader → Splitter → HF Embeddings → FAISS → Chain
│   └── web_search.py                    # Custom tool (@tool) + Tavily API + bind_tools
├── practice/
│   ├── 1_Langchain_quickstart.py        # Streamlit: ถาม-ตอบกับ LLM
│   ├── 2_Langchain_PromptTemplate.py    # Streamlit: สร้าง blog outline ด้วย prompt template
│   ├── 3_Langchain_search.py            # Streamlit: search agent (Google Serper + ReAct)
│   └── streamlit                        # ไฟล์ว่าง (ไม่ได้ใช้งาน)
├── example_code/
│   └── data.txt                         # ข้อมูลบริษัทสมมติสำหรับทดลอง RAG
├── .env                                 # (ต้องสร้างเอง) API keys และชื่อโมเดล
├── .gitignore
└── README.md
```

## วิธีใช้งาน

### 1. สร้าง virtual environment และติดตั้ง dependencies

```bash
cd LangChain-OpenRouter
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install "langchain<1.0" "langchain-community<1.0" "langchain-openai<1.0" \
            python-dotenv streamlit requests jupyter \
            sentence-transformers faiss-cpu
```

> โค้ดใช้ API ของ LangChain 0.x (`langchain.prompts`, `langchain.schema`, `langchain.text_splitter`, `initialize_agent`) ซึ่งถูกย้ายหรือลบออกใน LangChain 1.0 จึงควรติดตั้งเวอร์ชันต่ำกว่า 1.0

### 2. ตั้งค่า `.env`

สร้างไฟล์ `.env` ตามหัวข้อ [Environment Variables](#environment-variables)

### 3. รันบทเรียน (Notebook)

```bash
jupyter notebook tutorial/
```

- `llm_chain.ipynb` — รันทีละ cell ได้ทันที
- `rag.ipynb` — notebook โหลด `"data.txt"` จากโฟลเดอร์ปัจจุบัน ให้คัดลอก `example_code/data.txt` ไปไว้ใน `tutorial/` ก่อน (หรือแก้ path เป็น `"../example_code/data.txt"`) ครั้งแรกที่รันจะดาวน์โหลดโมเดล embedding จาก Hugging Face

### 4. รัน Tool Calling

```bash
python tutorial/web_search.py
```

### 5. รันแอป Streamlit

```bash
cd practice
streamlit run 1_Langchain_quickstart.py
streamlit run 2_Langchain_PromptTemplate.py
streamlit run 3_Langchain_search.py
```

เปิดเบราว์เซอร์ที่ <http://localhost:8501>

## หมายเหตุและสิ่งที่ควรพัฒนาต่อ

### ความปลอดภัย

- ไฟล์ `.env` ที่มี API keys ถูกกันไว้ใน [`.gitignore`](.gitignore) แล้ว (รวมถึง `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.streamlit/secrets.toml` และไฟล์ FAISS index) อย่า commit API key ลงในโค้ดโดยตรง

### ปัญหาที่พบในโค้ด

- **`2_Langchain_PromptTemplate.py`** — ส่ง `SystemMessage("... about {topic}.")` เข้า `ChatPromptTemplate` โดยตรง ซึ่ง message object จะไม่ถูกแทนค่าตัวแปร `{topic}` จึงถูกส่งไปตามตัวอักษร ควรใช้ tuple แทน เช่น `("system", "... about {topic}.")` และ `("human", "{topic}")`
- **`web_search.py`** — `bind_tools()` ทำให้โมเดล *ขอ* เรียก tool (อยู่ใน `response.tool_calls`) แต่โค้ดยังไม่ได้ **รัน tool และส่งผลกลับให้โมเดล** จึงยังไม่ได้คำตอบจากการค้นหาจริง (`response.content` อาจว่าง) ควรเพิ่ม loop เรียก tool แล้วส่ง `ToolMessage` กลับ หรือใช้ agent อย่าง `create_react_agent` ของ LangGraph นอกจากนี้ Tavily API ใช้พารามิเตอร์ `max_results` ไม่ใช่ `num_results`
- **`rag.ipynb`** — path `data.txt` ไม่ตรงกับตำแหน่งไฟล์จริง (`example_code/data.txt`)
- **ชื่อตัวแปร API key ไม่สม่ำเสมอ** — ไฟล์ส่วนใหญ่ใช้ `OPENAI_API_KEY` แต่ `web_search.py` ใช้ `OPENROUTER_API_KEY` ควรเลือกใช้ชื่อเดียว
- **`practice/streamlit`** เป็นไฟล์ว่าง ลบทิ้งได้

### API ที่ deprecated

| ที่ใช้อยู่ | ควรเปลี่ยนเป็น |
| --- | --- |
| `from langchain.prompts import ...` | `from langchain_core.prompts import ...` |
| `from langchain.schema import HumanMessage, SystemMessage` | `from langchain_core.messages import ...` |
| `from langchain.schema.runnable import RunnablePassthrough` | `from langchain_core.runnables import RunnablePassthrough` |
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` |
| `from langchain.utilities import GoogleSerperAPIWrapper` | `from langchain_community.utilities import GoogleSerperAPIWrapper` |
| `from langchain_community.embeddings import HuggingFaceEmbeddings` | `from langchain_huggingface import HuggingFaceEmbeddings` |
| `initialize_agent(..., AgentType.ZERO_SHOT_REACT_DESCRIPTION)` + `agent.run()` | `create_react_agent` (LangGraph) + `.invoke()` |

### แนวทางพัฒนาต่อ

- บันทึก FAISS index ลงดิสก์ (`vectorstore.save_local()`) เพื่อไม่ต้องสร้างใหม่ทุกครั้ง
- ทำ RAG เป็นแอป Streamlit ให้อัปโหลดเอกสาร (PDF/TXT) และถาม-ตอบได้
- ใช้ embedding model ที่รองรับภาษาไทยได้ดีกว่า เช่น `intfloat/multilingual-e5-base` หรือ `BAAI/bge-m3` (`all-MiniLM-L6-v2` ฝึกมากับภาษาอังกฤษเป็นหลัก)
- แสดงผลแบบ streaming ใน Streamlit ด้วย `st.write_stream()`
- เพิ่ม `requirements.txt`
