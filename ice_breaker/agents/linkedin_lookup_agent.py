from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from constants import MODEL_OPENAI_GPT_4O_MINI, PROMPT_HWCHASE17_REACT
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str):
    llm = ChatOpenAI(temperature=0, model_name=MODEL_OPENAI_GPT_4O_MINI)

    template = """given the full name {name_of_person} I want you to get the link to their LinkedIn profile page. 
    Your answer must contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google for LinkedIn Profile Page",
            func=get_profile_url_tavily,
            description="Useful when you need to get the LinkedIn Profile page URL",
        )
    ]

    react_prompt = hub.pull(PROMPT_HWCHASE17_REACT)
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url
