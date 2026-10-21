import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import initialize_agent
from langchain_classic.agents import AgentType
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

st.sidebar.title("settings")
api_key=st.sidebar.text_input("Enter ur api_key:",type="password")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="search")
llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-20b")
tools=[search,arxiv,wiki]
# 3. Prompt Template for Tool Calling Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant with access to web search, ArXiv, and Wikipedia tools. Always use tools when real-time information or factual verification is required."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Construct Agent and Executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 5. Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything, and I can search DuckDuckGo, Wikipedia, or Arxiv."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. Chat Input & Execution
if user_prompt := st.chat_input(placeholder="What is recent in AI?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_executor.invoke(
            {"input": user_prompt},
            config={"callbacks": [st_cb]}
        )
        output_text = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.write(output_text)




