import os
import json
import datetime
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from llama_index.core import Document

st.set_page_config(page_title="🦙 LlamaIndex Research Agent", layout="wide")

# Force explicit path to .env file relative to working directory
env_path = Path('.env').resolve()
load_dotenv(dotenv_path=env_path, override=True)

gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

st.sidebar.title("⚙️ Agent Settings")
if gemini_key and tavily_key:
    st.sidebar.success("✅ API Keys Loaded")
else:
    st.sidebar.error("❌ API Keys Missing in .env")
    st.sidebar.write(f"Looking at path: `{env_path}`")

@st.cache_resource
def get_clients(g_key, t_key):
    return genai.Client(api_key=g_key), TavilyClient(api_key=t_key)

st.title("🦙 LlamaIndex Autonomous Research Agent")

topic = st.text_input("Research Topic / Query:", placeholder="e.g., PAK VS SL TEST SERIES 2026")
start_btn = st.button("🚀 Run LlamaIndex Agent", type="primary")

if start_btn:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    elif not gemini_key or not tavily_key:
        st.error("API keys missing from .env file.")
    else:
        try:
            gemini_client, tavily_client = get_clients(gemini_key, tavily_key)

            with st.status("🧠 Decomposing Topic with Gemini...", expanded=True) as status:
                planner_prompt = f'Break down "{topic}" into 3 specific web search queries. Return ONLY a valid JSON array of 3 strings.'
                planner_res = gemini_client.models.generate_content(model="gemini-3.5-flash-lite", contents=planner_prompt)
                sub_queries = json.loads(planner_res.text.strip().replace("```json", "").replace("```", ""))
                status.update(label="✅ Query Decomposition Complete", state="complete")

            with st.status("🌐 Building LlamaIndex Documents...", expanded=True) as status:
                all_documents = []
                for idx, q in enumerate(sub_queries, 1):
                    s_res = tavily_client.search(query=q, max_results=2, search_depth="basic")
                    for res in s_res.get("results", []):
                        all_documents.append(Document(text=res.get("content", ""), extra_info={"url": res.get("url", ""), "sub_query": q}))
                status.update(label="✅ Ingestion Complete", state="complete")

            with st.status("🤖 Synthesizing Final Report...", expanded=True) as status:
                combined_context = "\n\n---\n\n".join([f"Query: {doc.extra_info.get('sub_query')}\nSource: {doc.extra_info.get('url')}\nContent: {doc.text}" for doc in all_documents])
                writer_prompt = f'Synthesize this into a research report on "{topic}":\n\n{combined_context}'
                final_res = gemini_client.models.generate_content(model="gemini-3.5-flash-lite", contents=writer_prompt)
                status.update(label="✅ Synthesis Complete", state="complete")

            report_text = final_res.text
            filename = f"llamaindex_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_text)

            st.success(f"💾 Saved as: `{filename}`")
            st.markdown(report_text)
            st.download_button("📥 Download Report (.md)", report_text, file_name=filename)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
