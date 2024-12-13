from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by its characters"""

    # Strip away all non alphanumeric characters
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Couldn't find tool with name: {tool_name}")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    tool_names = ", ".join([t.name for t in tools])

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=tool_names
    )

    # llm = ChatOpenAI(temperature=0, model_kwargs={"stop": ["\nObservation", "Observation"]})
    llm = ChatOpenAI(temperature=0, stop=["\nObservation"], model_name="gpt-4o-mini")
    # llm = ChatOllama(stop=["\nObservation"], model="llama3.2")
    intermediate_steps = []


    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
            }
            | prompt
            | llm
            | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of DOG in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of DOG in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
