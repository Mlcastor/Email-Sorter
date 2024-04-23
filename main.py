import os
from dotenv import load_dotenv
import json

from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools


search_tool = DuckDuckGoSearchRun()

load_dotenv()

# Create a new ChatGroq instance
GROQ_LLM = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192")
