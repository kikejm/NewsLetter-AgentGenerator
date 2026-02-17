import os
import functools
import re
from typing import Annotated, Literal, TypedDict
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Configuración Visual ---
st.set_page_config(page_title="News Writer", page_icon="📝", layout="centered")

# Ocultar elementos por defecto de Streamlit para una vista más limpia
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📝 Redactor IA")

# --- Sidebar Minimalista ---
with st.sidebar:
    st.header("Configuración")
    google_api_key = st.text_input("Google API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")

# --- Definiciones del Grafo ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Prompts optimizados para evitar ruido en el output
SEARCH_TEMPLATE = """You are a research assistant. Search for news relevant to the user's request. 
If you find enough info, stop searching. Do NOT write the article."""

OUTLINER_TEMPLATE = """Create a structured outline for the article based on the provided search results."""

WRITER_TEMPLATE = """Write the final article based on the outline.
STRICT FORMAT REQUIRED:
TITLE: <Insert Title Here>
BODY: <Insert Article Content Here>

IMPORTANT: Do not include markdown code blocks (like ```markdown). Do not include any introductory text. Just the Title and Body."""

# --- Lógica Core ---

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "outliner"

def build_graph(google_key, tavily_key):
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)
    tools = [TavilySearchResults(max_results=3)]

    search_agent = create_agent(llm, tools, SEARCH_TEMPLATE)
    outliner_agent = create_agent(llm, [], OUTLINER_TEMPLATE)
    writer_agent = create_agent(llm, [], WRITER_TEMPLATE)

    workflow = StateGraph(AgentState)

    workflow.add_node("search", functools.partial(agent_node, agent=search_agent, name="Search"))
    workflow.add_node("outliner", functools.partial(agent_node, agent=outliner_agent, name="Outliner"))
    workflow.add_node("writer", functools.partial(agent_node, agent=writer_agent, name="Writer"))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

def clean_and_parse_output(raw_text):
    """Limpia el output del LLM para mostrar solo título y texto formateado."""
    # Eliminar bloques de código markdown si existen
    text = re.sub(r'```[a-zA-Z]*', '', raw_text).strip()
    
    title = "Artículo Generado"
    body = text

    # Intentar separar por TITLE y BODY
    if "TITLE:" in text and "BODY:" in text:
        parts = text.split("BODY:")
        title_part = parts[0].replace("TITLE:", "").strip()
        body_part = parts[1].strip()
        return title_part, body_part
    
    return title, body

# --- Interfaz de Usuario ---

topic = st.text_input("¿Sobre qué quieres que escriba?", placeholder="Ej: Futuro de la IA en 2025")

if st.button("Generar") and topic:
    if not google_api_key or not tavily_api_key:
        st.error("Faltan las API Keys.")
        st.stop()

    with st.spinner("Investigando y escribiendo..."):
        try:
            app = build_graph(google_api_key, tavily_api_key)
            final_state = app.invoke({"messages": [HumanMessage(content=topic)]})
            
            # Obtener el último mensaje independientemente del tipo, asumiendo que es el del Writer
            last_message = final_state["messages"][-1]
            raw_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Limpieza y renderizado
            title, body = clean_and_parse_output(raw_content)
            
            st.divider()
            st.header(title)
            st.markdown(body)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")