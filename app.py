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

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Redactor IA · News Writer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&display=swap');

/* Fondo y base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F5F0E8;
    color: #1A1A1A;
}

[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    color: #F5F0E8;
}

[data-testid="stSidebar"] * {
    color: #F5F0E8 !important;
}

[data-testid="stSidebar"] input {
    background-color: #2A2A2A !important;
    border: 1px solid #444 !important;
    color: #F5F0E8 !important;
    border-radius: 4px !important;
}

/* Título principal */
.masthead {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: #1A1A1A;
    border-top: 4px solid #1A1A1A;
    border-bottom: 2px solid #1A1A1A;
    padding: 0.5rem 0;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}

.masthead-sub {
    font-family: 'Source Serif 4', serif;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 2rem;
}

/* Artículo generado */
.article-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.2;
    color: #1A1A1A;
    margin: 1rem 0 0.5rem 0;
    border-top: 3px solid #1A1A1A;
    padding-top: 1rem;
}

.article-body {
    font-family: 'Source Serif 4', serif;
    font-size: 1.1rem;
    font-weight: 300;
    line-height: 1.9;
    color: #2A2A2A;
    max-width: 720px;
    margin-top: 1rem;
}

.article-body p {
    margin-bottom: 1.2rem;
}

.article-meta {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: #888;
    border-left: 3px solid #C8A96E;
    padding-left: 0.75rem;
    margin-bottom: 1.5rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    font-family: monospace;
    font-size: 0.75rem;
    background: #1A1A1A;
    color: #C8A96E;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
}

/* Divider ornamental */
.ornament {
    text-align: center;
    color: #C8A96E;
    font-size: 1.2rem;
    letter-spacing: 8px;
    margin: 1.5rem 0;
}

/* Input styling */
[data-testid="stTextInput"] input {
    border: 1px solid #1A1A1A !important;
    border-radius: 0 !important;
    background: #FAF7F2 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 1rem !important;
    padding: 0.6rem !important;
}

/* Botón */
[data-testid="stButton"] > button {
    background-color: #1A1A1A !important;
    color: #F5F0E8 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s !important;
}

[data-testid="stButton"] > button:hover {
    background-color: #C8A96E !important;
    color: #1A1A1A !important;
}

/* Sidebar labels */
.sidebar-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-top: 1.2rem;
    margin-bottom: 0.2rem;
    font-family: monospace;
}

/* Info boxes */
.info-box {
    border-left: 3px solid #C8A96E;
    background: #FAF0DC;
    padding: 0.75rem 1rem;
    font-family: 'Source Serif 4', serif;
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* Ocultar elementos de Streamlit */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CABECERA TIPO PERIÓDICO
# ─────────────────────────────────────────────
st.markdown('<div class="masthead">📰 Redactor IA</div>', unsafe_allow_html=True)
st.markdown('<div class="masthead-sub">Inteligencia Artificial · Redacción Automatizada · Powered by Gemini & Tavily</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    st.markdown('<div class="sidebar-label">🔑 Google Gemini API Key</div>', unsafe_allow_html=True)
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">🔍 Tavily Search API Key</div>', unsafe_allow_html=True)
    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🧠 Flujo del Agente")
    st.markdown("""
    ```
    🔍 Search Agent
         ↓  (itera si necesita más datos)
    🛠️  Tool Node (Tavily)
         ↓
    📋 Outliner Agent
         ↓
    ✍️  Writer Agent
         ↓
       Artículo
    ```
    """)
    st.markdown("---")
    st.markdown("**Modelo:** `gemini-2.5-flash`")
    st.markdown("**Búsquedas:** máx. 3 resultados/ciclo")

# ─────────────────────────────────────────────
# LANGGRAPH — DEFINICIONES
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SEARCH_TEMPLATE = """You are a research assistant. Your job is to search the web for accurate, recent information relevant to the user's topic.
Use the search tool as many times as needed to gather enough information.
Once you have gathered sufficient data, stop searching. Do NOT write the article yourself."""

OUTLINER_TEMPLATE = """You are an expert editorial editor. Based on the search results in the conversation, 
create a clear and structured outline for a well-written news article. 
Include: headline idea, key sections, and bullet points of the main facts to cover per section."""

WRITER_TEMPLATE = """You are an expert journalist. Write a compelling, well-structured article based on the outline and research provided.

STRICT FORMAT — output ONLY the following, nothing else:

TITLE: <The article headline here>
BODY: <Full article content here, written in clear paragraphs>

Rules:
- Do NOT include markdown fences (```).
- Do NOT include any intro text like "Here is your article".
- Do NOT include the words TITLE or BODY in the article text itself.
- Write in a professional yet engaging journalistic style.
- Minimum 4 paragraphs."""


def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


def agent_node(state: AgentState, agent, name: str):
    result = agent.invoke(state)
    result.name = name
    return {"messages": [result]}


def should_search(state: AgentState) -> Literal["tools", "outliner"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "outliner"


@st.cache_resource(show_spinner=False)
def build_graph(google_key: str, tavily_key: str):
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    tools = [TavilySearchResults(max_results=3)]

    search_agent = create_agent(llm, tools, SEARCH_TEMPLATE)
    outliner_agent = create_agent(llm, [], OUTLINER_TEMPLATE)
    writer_agent = create_agent(llm, [], WRITER_TEMPLATE)

    workflow = StateGraph(AgentState)

    workflow.add_node("search", functools.partial(agent_node, agent=search_agent, name="Search"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", functools.partial(agent_node, agent=outliner_agent, name="Outliner"))
    workflow.add_node("writer", functools.partial(agent_node, agent=writer_agent, name="Writer"))

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search, {"tools": "tools", "outliner": "outliner"})
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


def clean_and_parse_output(raw_text: str) -> tuple[str, str]:
    """Extrae título y cuerpo del output del Writer Agent."""
    # Eliminar bloques de código si el LLM los incluyó igualmente
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text).strip()
    text = text.replace("```", "").strip()

    if "TITLE:" in text and "BODY:" in text:
        parts = text.split("BODY:", 1)
        title = parts[0].replace("TITLE:", "").strip()
        body = parts[1].strip()
        return title, body

    # Fallback: primer párrafo como título
    lines = text.strip().splitlines()
    title = lines[0].strip() if lines else "Artículo Generado"
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return title, body


def format_body_as_html(body: str) -> str:
    """Convierte saltos de línea en párrafos HTML."""
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])

with col_input:
    topic = st.text_input(
        "Tema del artículo",
        placeholder="Ej: El impacto de la IA generativa en el periodismo en 2025",
        label_visibility="collapsed",
    )

with col_btn:
    generate = st.button("GENERAR", use_container_width=True)

st.markdown(
    '<div class="info-box">💡 Escribe un tema, evento o pregunta y el agente investigará, creará un esquema y redactará un artículo completo.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# GENERACIÓN
# ─────────────────────────────────────────────

if generate and topic:
    if not google_api_key or not tavily_api_key:
        st.error("⚠️ Introduce ambas API Keys en el panel lateral para continuar.")
        st.stop()

    # Progress steps UI
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown('<div class="status-badge">⟳ PROCESANDO</div>', unsafe_allow_html=True)
        step1 = st.empty()
        step2 = st.empty()
        step3 = st.empty()
        step4 = st.empty()

    step1.info("🔍 **Buscando información** en la web...")

    try:
        app = build_graph(google_api_key, tavily_api_key)

        # Streaming del grafo para actualizar pasos
        title_out, body_out = "Artículo Generado", ""
        agent_sequence = []

        for event in app.stream(
            {"messages": [HumanMessage(content=topic)]},
            stream_mode="values",
        ):
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                name = getattr(last, "name", None)

                if name == "Search" and "outliner" not in agent_sequence:
                    if "search" not in agent_sequence:
                        agent_sequence.append("search")
                        step1.success("✅ **Búsqueda completada.**")
                        step2.info("📋 **Creando esquema** del artículo...")

                if name == "Outliner" and "outliner" not in agent_sequence:
                    agent_sequence.append("outliner")
                    step2.success("✅ **Esquema listo.**")
                    step3.info("✍️ **Redactando el artículo final...**")

                if name == "Writer" and "writer" not in agent_sequence:
                    agent_sequence.append("writer")
                    raw = last.content if hasattr(last, "content") else str(last)
                    title_out, body_out = clean_and_parse_output(raw)
                    step3.success("✅ **Artículo redactado.**")
                    step4.success("🎉 ¡Listo!")

        # Limpiar progress
        progress_placeholder.empty()

        if not body_out:
            st.warning("El agente no devolvió contenido. Intenta con un tema diferente.")
            st.stop()

        # ── Renderizado del artículo ──
        st.markdown('<div class="ornament">— ✦ —</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="article-title">{title_out}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="article-meta">Generado por Redactor IA · Gemini 2.5 Flash · Fuentes: Tavily Search</div>',
            unsafe_allow_html=True,
        )

        html_body = format_body_as_html(body_out)
        st.markdown(f'<div class="article-body">{html_body}</div>', unsafe_allow_html=True)
        st.markdown('<div class="ornament">— ✦ —</div>', unsafe_allow_html=True)

        # Opción de copiar texto plano
        with st.expander("📋 Ver texto plano para copiar"):
            st.text_area("Artículo completo", value=f"{title_out}\n\n{body_out}", height=300)

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ Error durante la generación: {str(e)}")
        st.info("Verifica que tus API Keys sean correctas y tengas conexión a internet.")

elif generate and not topic:
    st.warning("Por favor, escribe un tema antes de generar.")