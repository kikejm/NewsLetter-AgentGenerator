import os
import functools
from typing import Annotated, Literal, TypedDict
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Configuración de la Página ---
st.set_page_config(page_title="News Writer Agent", page_icon="📰", layout="centered")
st.title("📰 Generador de Artículos con LangGraph")

# --- Sidebar: Credenciales ---
with st.sidebar:
    st.header("Configuración")
    google_api_key = st.text_input("Google API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")


# --- Definición de Tipos y Prompts ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SEARCH_TEMPLATE = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node."""

OUTLINER_TEMPLATE = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline for the article."""

WRITER_TEMPLATE = """Your job is to write an article, do it in this format:
TITLE: <title>
BODY: <body>
NOTE: Do not copy the outline. You need to write the article with the info provided by the outline.
```"""

# --- Lógica del Grafo (Backend) ---

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

def build_app(google_key, tavily_key):
    # Configuración de entorno segura por ejecución
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    # Inicialización de recursos
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    tools = [TavilySearchResults(max_results=5)]

    # Definición de Agentes
    search_agent = create_agent(llm, tools, SEARCH_TEMPLATE)
    outliner_agent = create_agent(llm, [], OUTLINER_TEMPLATE)
    writer_agent = create_agent(llm, [], WRITER_TEMPLATE)

    # Nodos
    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")
    tool_node = ToolNode(tools)

    # Estructura del Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# --- Interfaz de Usuario (Frontend) ---

topic = st.text_input("Tema del artículo:", placeholder="Ej: Impacto de la IA en la medicina 2024")

if st.button("Redactar Artículo"):
    if not google_api_key or not tavily_api_key:
        st.error("⚠️ Faltan las API Keys en la configuración.")
        st.stop()
    
    if not topic:
        st.warning("⚠️ Debes introducir un tema.")
        st.stop()

    # Contenedor para feedback visual mínimo
    with st.spinner("🕵️‍♂️ Los agentes están investigando y redactando... (Esto puede tomar unos segundos)"):
        try:
            # Construir y compilar grafo
            app = build_app(google_api_key, tavily_api_key)
            
            # Ejecución completa síncrona (invoke en lugar de stream)
            inputs = {"messages": [HumanMessage(content=topic)]}
            final_state = app.invoke(inputs)
            
            # Extracción del último mensaje (Output del Writer)
            last_message = final_state["messages"][-1]
            
            if isinstance(last_message, AIMessage) and last_message.content:
                st.success("¡Artículo generado!")
                st.markdown("---")
                st.markdown(last_message.content)
            else:
                st.error("El flujo finalizó pero no devolvió contenido válido.")
                
        except Exception as e:
            st.error(f"Error crítico en la ejecución: {str(e)}")