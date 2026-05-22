# app.py
"""
Streamlit demo app for the OpenAgenda RAG chatbot.

Run:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.rag.rag_pipeline import SYSTEM_PROMPT, USER_PROMPT, _format_context


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Puls-Events RAG Assistant",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 Puls-Events RAG Assistant")
st.write("Ask questions about cultural events indexed from OpenAgenda.")


# ------------------------------------------------------------
# Environment and settings
# ------------------------------------------------------------

load_dotenv(find_dotenv(usecwd=True))

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("MISTRAL_API_KEY is missing. Add it to your .env file.")
    st.stop()


with st.sidebar:
    st.header("Settings")

    index_path = st.text_input("FAISS index path", value="data/index/faiss")
    k = st.slider("Number of retrieved chunks", min_value=3, max_value=10, value=5)
    model = st.selectbox(
        "Mistral model",
        options=["mistral-small-latest", "mistral-medium-latest"],
        index=0,
    )

    st.info(
        "If the FAISS index does not exist, build it first:\n\n"
        "`python -m src.index.build_faiss --city Paris --max-records 1000 --index-out data/index/faiss`"
    )


# ------------------------------------------------------------
# Load FAISS and Mistral
# ------------------------------------------------------------

@st.cache_resource
def load_vectorstore(index_path: str, api_key: str):
    """Load the FAISS index only once."""

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    return FAISS.load_local(
        index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm(model: str, api_key: str):
    """Load the Mistral chat model only once."""

    return ChatMistralAI(
        model=model,
        temperature=0.1,
        api_key=api_key,
    )


if not os.path.exists(os.path.join(index_path, "index.faiss")):
    st.error(f"FAISS index not found at: {index_path}")
    st.stop()

vectordb = load_vectorstore(index_path, api_key)
llm = load_llm(model, api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]
)


# ------------------------------------------------------------
# User question
# ------------------------------------------------------------

question = st.text_input(
    "Your question",
    placeholder="Example: Recommande-moi une activité culturelle pour des enfants à Paris.",
)

if st.button("Ask") and question.strip():
    with st.spinner("Searching events and generating answer..."):

        # 1. Retrieve relevant chunks from FAISS
        docs = vectordb.similarity_search(question, k=k)

        if not docs:
            st.warning("I don't know based on the available event data.")
            st.stop()

        # 2. Format retrieved chunks as context
        context = _format_context(docs)

        # 3. Generate answer with Mistral
        messages = prompt.format_messages(question=question, context=context)
        response = llm.invoke(messages)

    st.subheader("Answer")
    st.write(response.content)

    st.subheader("Retrieved sources")

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}

        with st.expander(f"Source {i}: {meta.get('title', 'Untitled event')}"):
            st.write(f"**City:** {meta.get('city', '')}")
            st.write(f"**Venue:** {meta.get('venue', '')}")
            st.write(f"**Start:** {meta.get('start_utc', '')}")
            st.write(f"**End:** {meta.get('end_utc', '')}")

            url = meta.get("url", "")
            if url:
                st.write(f"**URL:** {url}")

            st.write("**Snippet:**")
            st.write(doc.page_content[:800])