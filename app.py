import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup

import requests
import json

from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

#1. Tool for search
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

#search("What can I do with OpenAI APIs?")  

#2. Tool for web scraping
def scrape_website(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content
    #objective is the original objective & task that user give to the agent, url is the url

    print("Scraping website...")
    #Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    #Define the data to be sent in the request
    data = {
        "url": url
    }

    #Convert Python object into JSON string
    data_json = json.dumps(data)

    #Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    #Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

#2b. Summarize function: function for summarizing results from web scraper

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])


    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

#2c. Tool to be implemente in langchain agent


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")




#3. Create lanchain agent with the tools above

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""    
    You are a world-class researcher with a specialty in art and art collectors. Your mission is to produce accurate, fact-based, and up-to-date profiles while being sensitive to dates and distinguishing between current and past events. 
    
    Your goal is to provide a comprehensive, yet concise profile of the sender and their associated company, focusing on their involvement in the art world.
    
    Please make sure you complete the objective above with the following rules:
    1/ Provide me with detailed information about the sender and company from this email.
    2/ I need to know more about the history, reputation, and main activities of the sender of this email.
    3/ Delve into the sender's background, especially their financial stability, past art purchases, and any participation in high-end art auctions.
    4/ Seek testimonials, reviews, or mentions from other art sellers or collaborators.
    5/ You should do enough research to gather as much information as possible about the objective.
    6/ If there are URLs of relevant links & articles, you will scrape it to gather more information.
    7/ If there are no results about the sender of the email in English, scrape and search results in French, Spanish, Italian and Russian, translate those results into English and summarize them.
    8/ After scraping & search, you should think "is there any new things I should search & scraping based on the data I collected to increase research quality?" If the answer is yes, continue; But don't do this more than 3 iterations.
    9/ You should only write facts & data that you have gathered, and avoid making assumptions.
    10/ Keep the information fluid, narrative-like, and omit sections where specific information isn't available.
    11/ In the final output, You should include all reference data & links to back up your research.
    12/ Additionally, determine if the sender is a high net-worth individual or a distinguished art collector.
    13/ Provide evidence or sources that support your findings.
    """
)




agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


#4 Use streamlit to create web app

# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)

#         result = agent({"input": query})

#         st.info(result['output'])

# if __name__ == '__main__':
#     main()

#5 Set this as an API endpoint via FastAPI (instead of using streamlit)

app = FastAPI()

class Query(BaseModel):
    query: str

def parse_content_to_dict(content: str):
    sections = [
        "Sender Profile", "Background", "Company Profile", 
        "Reputation and History", "Financial Stability", 
        "Art Purchases and Auction Participation", "Testimonials and Reviews", 
        "High Net-Worth Individual or Distinguished Art Collector", "References"
    ]

    # Initialize the dictionary with default values
    data = {section: "Not Available" for section in sections}

    for i, section in enumerate(sections):
        start_idx = content.find(section)
        if start_idx != -1:  # If the section is found in the content
            if i < len(sections) - 1:
                # Get the start index of the next section
                next_section_idx = content.find(sections[i+1])
                data[section] = content[start_idx+len(section)+1:next_section_idx].strip()  # +1 to skip colon
            else:
                # For the last section, take everything after the section title
                data[section] = content[start_idx+len(section)+1:].strip()  # +1 to skip colon

    return data

@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']

    return parse_content_to_dict(actual_content)