from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
import os
from dotenv import load_dotenv
from langchain_experimental.cpal.templates.univariate.query import template
from langchain_openai.llms.base import OpenAI
from langchain_ollama.llms import OllamaLLM

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["GORQ_API_KEY"]=os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.0,api_key=os.environ["OPENAI_API_KEY"])
# llm = OllamaLLM(model="llama3.2")
print("llm :{}".format(llm))

prompt = PromptTemplate(
   input_variables = ["user_text"],
   template="tell me about {user_text} in just 5 words."
)
chain = prompt|llm
resp = chain.invoke({"user_text":"India"})
print(resp)