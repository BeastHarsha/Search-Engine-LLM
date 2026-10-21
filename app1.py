from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.agents import AgentType
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
import streamlit as st
import os

st.sidebar.title("settings")
api_key=st.sidebar.text_input("Enter your api key:")
if api_key:
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
    wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

    ddg=DuckDuckGoSearchRun(name="search")
    pdfloader=PyPDFLoader('LLM.pdf')

    documents=pdfloader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size
    =1000,
        chunk_overlap=150)
    text_splits=text_splitter.split_documents(documents)
    if not text_splits:
        st.error(
        "No extractable text was found in this PDF. If this is a scanned document or image-only PDF, standard text extraction cannot read it."
    )
    embeddings=OllamaEmbeddings(model="nomic-embed-text")
    vectordb=FAISS.from_documents(text_splits,embeddings)
    retriever=vectordb.as_retriever()
    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-20b")

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system","behave like an ai assistant and do the needful to the user"),
            ("user","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="Search through the uploaded PDF document",
    )
    tools = [arxiv, wiki, ddg, retriever_tool]
    agent=create_tool_calling_agent(llm,tools,prompt)
    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role":"assistant","content":"what can i do for you> i will search your response from arxiv,duckduckgo or wikipedia"}
        ]

    for msg in st.session_state['messages']:
        st.chat_message(msg['role']).write(msg['content'])

    if user_input:=st.chat_input(placeholder="what is recent ai news"):
        st.session_state['messages']=[{"role":"user","content":user_input}]
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agent_executor.invoke({"input":user_input,
            "config":{"callbacks":[st_cb]}})
            output_text=response['output']
            st.session_state.messages.append({"role":"assistant","content":output_text})
            st.write(output_text)









