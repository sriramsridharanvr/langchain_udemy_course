from typing import Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from constants import MODEL_OPENAI_GPT_4O_MINI
from output_parsers import summary_parser, Summary
from third_parties.linkedin import scrape_linkedin_profile


# from langchain_ollama import ChatOllama


def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url, mock=True)
    summary_template = """
            Given the LinkedIn information {information} about a person, I want you to create:
            1. a short summary
            2. two interesting facts about them
            
            \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature=0, model_name=MODEL_OPENAI_GPT_4O_MINI)
    chain = summary_prompt_template | llm | summary_parser
    res:Summary = chain.invoke(input={"information": linkedin_data})
    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    ice_break_with("Andrew Ng")
