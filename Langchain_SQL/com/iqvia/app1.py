from gc import callbacks

import dotenv
import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.llms.openai import OpenAIChat
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from openai import api_key
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_openai import OpenAI,ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["GORQ_API_KEY"]=os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

st.title("Chat with SQL database.")

@st.cache_resource(ttl="1h")
def connect_database():
    uri = f"mysql+pymysql://ajay:ajay@localhost:3306/ajay"
    db = SQLDatabase.from_uri(uri,schema="ajay")
    return db

db = connect_database()

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name='gpt-3.5-turbo')
sql_toolkit = SQLDatabaseToolkit(db=db,llm=llm)
agent = create_sql_agent(llm=llm,toolkit=sql_toolkit,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,agent_executor_kwargs={"handle_parsing_errors":True})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

user_query = st.chat_input(placeholder="ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["output"]
        })
        st.write(response["output"])
