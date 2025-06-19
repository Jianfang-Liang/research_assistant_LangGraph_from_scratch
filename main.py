
# pip install -U langgraph langgraph-supervisor langchain-tavily "langchain[openai]"

# Create research agent
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

search = TavilySearch(max=3)

research_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks.\n"
        "- Do not perform analysis, interpretation, or translation.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name= "research_agent",
)

# Create analysis agent
analysis_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt=(
        "You are an analysis agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Your role is to analyze and summarize information collected by the research agent.\n"
        "- Focus on extracting key insights, comparing data points, identifying trends or contradictions.\n"
        "- Respond ONLY with your analytical conclusions, do NOT include any extra commentary or explanations.\n"
        "- Do not perform translation.\n"
        "- After you're done, respond directly to the supervisor."
    ),
    name="analysis_agent"
)

# Create translation agent
translation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt=(
        "You are a translation agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Your task is to translate the content provided by the analysis agent into the target language specified by the supervisor.\n"
        "- Translate accurately and fluently while preserving the original meaning and tone.\n"
        "- Do NOT add any explanations, commentary, or formatting.\n"
        "- Do NOT perform any analysis or research.\n"
        "- If the target language is not specified, ask the supervisor for clarification.\n"
        "- After translating, respond directly to the supervisor with ONLY the translated content."        
    ),
    name="translation_agent",
)

# Set up agent communication -- implement handoffs via handoff tools
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None): 
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


# Handoffs
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_analysis_agent = create_handoff_tool(
    agent_name="analysis_agent",
    description="Assign task to an analysis agent.",
)

assign_to_translation_agent = create_handoff_tool(
    agent_name="translation_agent",
    description="Assign task to an translation agent.",
)

# Create supervisor agent
supervisor_agent = create_react_agent(
    model = "openai:gpt-4o-mini",
    tools = [assign_to_research_agent, assign_to_analysis_agent, assign_to_translation_agent], 
    prompt=(
        "You are a supervisor managing three agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a analysis agent. Assign analysis-related tasks to this agent\n"
        "- a translation agent. Responsible for translating only the outputs provided by the analysis agent after analysis is complete."
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)

# Create multi-agent graph
from langgraph.graph import END

supervisor = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent, destinations=("research_agent", "analysis_agent","translation_agent", END))
    .add_node(research_agent)
    .add_node(analysis_agent)
    .add_node(translation_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent","supervisor")
    .add_edge("analysis_agent","supervisor")
    .add_edge("translation_agent","supervisor")
    .compile()
)

# Create the graph png picture
from PIL import Image

image_bytes = supervisor.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(image_bytes)

from PIL import Image
img = Image.open("graph.png")
img.show()


# Run the test
from pretty_print_messages import pretty_print_messages
for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Please find recent applications of AI in education. Then analyze the main benefits. Finally, translate your analysis into Chinese."
                # Change your query here. Try to put in a prompt that will call for the 3 agents.
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]