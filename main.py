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


# Create the agents
class EmailAgents:
    def make_categorizer_agent():
        return Agent(
            role="""take in a email from a human that has emailed out company email address and categorize it into one of the following categories: \
            price_equiry - the email is asking for the price of a product or service \
            customer_complaint - used when someone is complaining about something \
            product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \
            customer_feedback - used when someone is giving feedback about a product or service \
            off_topic - used when the email is not related to any of the above categories""",
            backstory="""You are a master at understanding what a customer wants when they write an email and are able to categorize it in a useful way.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Email Categorizer Agent"),
        )
