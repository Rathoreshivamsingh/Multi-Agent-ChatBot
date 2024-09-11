from typing import Any,Dict,List
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.chains import LLMMathChain
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate
from url import run_llm,url
import logging
logging.basicConfig(level=logging.INFO)

WEATHER_API_KEY='77ea96943546e91d6dce1760331f2f02'

load_dotenv()

llm=ChatGroq(temperature=0, model="mixtral-8x7b-32768")
llm_math = LLMMathChain.from_llm(llm)



weather_tool = OpenWeatherMapAPIWrapper(openweathermap_api_key=WEATHER_API_KEY)


tools = [
        Tool(
             name="Calculator",
                func=llm_math.invoke,
                 description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only inputmath expressions."
        ),
        Tool(
                name="Weather",
                func=weather_tool.run,
                description="""Useful when you need to get the current weather information for any location. 
           Takes an input of the location (city name) and returns the current weather conditions, 
           including temperature, humidity, and weather description after fetching data from a weather API."""
        ),
        Tool(
    name="LLM URL Tool",
    func=lambda query: run_llm(query),
    description="This tool uses an LLM to answer questions based on a retrieval-augmented generation (RAG) model."
)
        
    ]


prompt_template = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "chat_history", "agent_scratchpad"],
    template="""
Answer the following questions as best you can. You have access to the following tools:

{tools}

The tools are categorized as follows:
- Calculator: For mathematical calculations.
- Weather: For current weather information based on a city name.
- URL Processor and Query Tool: For processing URLs and querying their content.

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


New input: {input}

Tool selection and response handling instructions:

{agent_scratchpad}
"""
)

tools_list = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
tool_names = ", ".join([tool.name for tool in tools])

prompt = prompt_template.partial(
    instructions="",
    tools=tools_list,
    tool_names=tool_names
)

grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatGroq(temperature=0, model="mixtral-8x7b-32768"),
        tools=tools,
       
    )
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True,handle_parsing_errors=True)


def handle_input(input_text: str,chat_history: List[Dict[str, Any]] = []):
    logging.info(f"Input received: {input_text}")
    """
    This function handles the input and determines whether it's a URL or a question,
    then invokes the appropriate tool using the LangChain agent.
    """
    
    # Define the tools and tool names as required
    tool_names = ", ".join([tool.name for tool in tools])
    tools_list = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    chat_history = []  # Initialize or load chat history as needed
    agent_scratchpad = ""  # Initialize or update as needed
    if input_text.startswith("http://") or input_text.startswith("https://"):
            embed= url(input_text)
            res = 'Now Ask Me the Question based on the url '
            return {"answer": res}
    else:
        input_data = {
            "input": input_text,  # Query input
            "tools": tools_list,
            "tool_names": tool_names,
            "chat_history": chat_history,
            "agent_scratchpad": agent_scratchpad,
        }
        try:
            response = grand_agent_executor.invoke(input={"input": input_data, "chat_history": chat_history})
            logging.info(f"Response received: {response}")
            if isinstance(response, dict) and 'output' in response:
                return {'answer': response['output']}
            else:
                return {"answer": "No response available."}
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {"answer": str(e)}