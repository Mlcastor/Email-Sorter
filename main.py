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
    def make_categorizer_agent(self):
        return Agent(
            role="Email Categorizer Agent",
            goal="""take in a email from a human that has emailed out company email address and categorize it into one of the following categories: \
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
    
    def make_researcher_agent(self):
        return Agent(
            role="Researcher Agent",
            goal="""take in a email from a human that has emailed out company email address and the category that the categorizer agent gave it
            and decide what information you need to search for for the email in a thoughtful and helpful way. 
            if you DONT think a search will help just reply 'NO SEARCH NEEDED'
            If you dont find any useful information in the search results, reply 'NO USEFUL RESEARCH FOUND'
            otherwise reply with the info you found that is useful for the email writer to write a helpful response to the email""",
            backstory="""You are a master at understanding what information our email writer needs to write a helpful response to the email""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Researcher Agent"),
        )
    
    def make_email_writer_agent(self):
        return Agent(
            role="Email Writer Agent",
            goal="""take in a email from a human that has emailed out company email address, the category \
                that the categorizer agent gave it and the information that the researcher agent found and write a helpful response to the email in a thoughtful and friendly way.
                
                If the customer email is "off_topic" then ask them questions to get more information.
                If the customer email is "price_equiry" then try to give them the price of the product or service.
                If the customer email is "customer_complaint" then try to assure we value them as a customer and try to solve their problem.
                If the customer email is "product_enquiry" then try to give them the information they are asking for.
                If the customer email is "customer_feedback" then try to assure we value them and make them feel heard.
                
                You never make up informations that hasn't been provided by the researcher or in the email.
                Always sign off the emails in appropriate manner and from Sarah, the customer service manager.""",
            backstory="""You are a master at synthesizing a variety of information and writing a helpful email that will address the customer's issues and provide them with helpful information."""
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Email Writer Agent"),
        )
        

class EmailTasks():
    def categorize_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided and categorize into \
                one of the following categories:
                price_equiry - used when someone is asking for the price of a product or service \
                customer_complaint - used when someone is complaining about something \
                product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \
                customer_feedback - used when someone is giving feedback about a product or service \
                off_topic - used when the email is not related to any of the above categories \
                
                EMAIL CONTENT:\n\n {email_content} \n\n
                Output a single category only.""",
            output_file=f"email_category.txt",
            agent=categorizer_agent
        )
    
    def research_info_for_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided and the category that the categorizer agent gave it and decide what information you need to search for for the email in a thoughtful and helpful way. 
            if you DONT think a search will help just reply 'NO SEARCH NEEDED'
            If you dont find any useful information in the search results, reply 'NO USEFUL RESEARCH FOUND'
            otherwise reply with the info you found that is useful for the email writer to write a helpful response to the email
            
            EMAIL CONTENT:\n\n {email_content} \n\n
            Only provide the info needed DONT write the email response""",
            expected_output="""A set of bullet points of useful info for the email writer \
            or clear instructions that no useful material was found""",
            context = [categorize_email],
            output_file=f"research_info.txt",
            agent=researcher_agent
        )
    
    def draft_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided and the category that the categorizer agent gave it \
            and the information that the researcher agent found to write a helpful response to the email in a thoughtful and friendly way. \
            
            Write a simple, polite and too the point email wich will respond to the customer's email. \
            if useful use the info provided by the researcher agent to write the email. \

            If no useful info was provided from the research specialist don't make up any information, don't try to guess. \
            
            EMAIL CONTENT:\n\n {email_content} \n\n
            Only provide the email response""",
            expected_output="""A well written email response to the customer that adress their issues and provide them with helpful information""",
            context = [categorize_email, research_info_for_email],
            output_file=f"email_response.txt",
            agent=email_writer_agent
        )

EMAIL = """Hi there, \n
I am emailing to say that I had a wonderful stay at your resort last week. \n

I really appreaciate what your staff did. \n
Thanks, \n
John Doe"""

# instanciate the agents
agents = EmailAgents()
tasks = EmailTasks()

#Agents
categorizer_agent = agents.make_categorizer_agent()
researcher_agent = agents.make_researcher_agent()
email_writer_agent = agents.make_email_writer_agent()

#Tasks
categorize_email = tasks.categorize_email(EMAIL)
research_info_for_email = tasks.research_info_for_email(EMAIL)
draft_email = tasks.draft_email(EMAIL)