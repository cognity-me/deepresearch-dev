# pylint: disable=line-too-long,useless-suppression
# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
DESCRIPTION:
    This sample demonstrates how to use Agent operations with the Deep Research tool from
    the Azure Agents service through the **asynchronous** Python client. Deep Research issues
    external Bing Search queries and invokes an LLM, so each run can take several minutes
    to complete.

    For more information see the Deep Research Tool document: https://aka.ms/agents-deep-research

USAGE:
    python sample_agents_deep_research_async.py

    Before running the sample:

    pip install azure-identity aiohttp
    pip install --pre azure-ai-projects

    Set these environment variables with your own values:
    1) PROJECT_ENDPOINT - The Azure AI Project endpoint, as found in the Overview
                          page of your Azure AI Foundry portal.
    2) MODEL_DEPLOYMENT_NAME - The deployment name of the arbitration AI model, as found under the "Name" column in
       the "Models + endpoints" tab in your Azure AI Foundry project.
    3) DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME - The deployment name of the Deep Research AI model, as found under the "Name" column in
       the "Models + endpoints" tab in your Azure AI Foundry project.
    4) BING_RESOURCE_NAME - The resource name of the Bing connection, you can find it in the "Connected resources" tab
       in the Management Center of your AI Foundry project.
"""

import asyncio
import os
from typing import Optional, List
# from datetime import datetime
from dotenv import load_dotenv
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    MessageRole,
    ThreadMessage,
    MessageTextContent,
    DeepResearchToolDefinition,
    DeepResearchDetails,
    DeepResearchBingGroundingConnection,
)
from azure.identity.aio import DefaultAzureCredential

load_dotenv()


async def fetch_and_print_new_agent_response(
    thread_id: str,
    agents_client: AgentsClient,
    last_message_id: Optional[str] = None,
) -> Optional[str]:
    response = await agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )

    if not response or response.id == last_message_id:
        return last_message_id

    print("\nAgent response:")
    print("\n".join(t.text.value for t in response.text_messages))

    # Print citation annotations (if any)
    for ann in response.url_citation_annotations:
        print(f"URL Citation: [{ann.url_citation.title}]({ann.url_citation.url})")

    return response.id


def print_messages_and_save_summary(messages: List[ThreadMessage], summary_filepath: str = "research_summary.md") -> None:
    """
    Prints messages from a thread and saves the last agent message to a summary file.

    :param messages: A list of ThreadMessage objects.
    :param summary_filepath: The path to the file where the summary will be saved.
    """
    last_agent_message = None
    for message in messages:
        print(f"{message.created_at.strftime('%Y-%m-%d %H:%M:%S')} - {message.role:>10}: ", end="")
        full_response = []
        for content_item in message.content:
            if isinstance(content_item, MessageTextContent):
                response_text = content_item.text.value
                annotations = content_item.text.annotations
                if annotations:
                    for annotation in annotations:
                        # MessageTextUriCitationAnnotation을 직접 임포트하지 않으므로,
                        # 객체가 url_citation 속성을 가지고 있는지 확인합니다.
                        if hasattr(annotation, 'url_citation') and hasattr(annotation.url_citation, 'title') and hasattr(annotation.url_citation, 'url'):
                            response_text = response_text.replace(
                                annotation.text,
                                f" [{annotation.url_citation.title}]({annotation.url_citation.url})"
                            )
                full_response.append(response_text)
                print(response_text, end="")

        print()
        if message.role == MessageRole.AGENT:
            last_agent_message = "\n".join(full_response)

    if last_agent_message:
        with open(summary_filepath, "w", encoding="utf-8") as f:
            f.write(last_agent_message)
        print(f"\nResearch summary saved to '{summary_filepath}'")


async def main() -> None:

    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

    bing_connection = await project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"])

    # Initialize a Deep Research tool with Bing Connection ID and Deep Research model deployment name
    deep_research_tool = DeepResearchToolDefinition(
        deep_research=DeepResearchDetails(
            deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
            deep_research_bing_grounding_connections=[
                DeepResearchBingGroundingConnection(connection_id=bing_connection.id)
            ],
        )
    )

    async with project_client:

        agents_client = project_client.agents

        # Create a new agent that has the Deep Research tool attached.
        # NOTE: To add Deep Research to an existing agent, fetch it with `get_agent(agent_id)` and then,
        # update the agent with the Deep Research tool.
        agent = await agents_client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="my-agent",
            instructions="You are a helpful Agent that assists in researching scientific topics.",
            tools=[deep_research_tool],
        )
        print(f"Created agent, ID: {agent.id}")

        # Create thread for communication
        thread = await agents_client.threads.create()
        print(f"Created thread, ID: {thread.id}")

        # Create message to thread
        message = await agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=(
                "Research the current state of studies on orca intelligence and orca language, "
                "including what is currently known about orcas' cognitive capabilities, "
                "communication systems and problem-solving reflected in recent publications in top thier scientific "
                "journals like Science, Nature and PNAS."
            ),
        )
        print(f"Created message, ID: {message.id}")

        print("Start processing the message... this may take a few minutes to finish. Be patient!")
        # Poll the run as long as run status is queued or in progress
        run = await agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)
        last_message_id: Optional[str] = None
        while run.status in ("queued", "in_progress"):
            await asyncio.sleep(1)
            run = await agents_client.runs.get(thread_id=thread.id, run_id=run.id)

            last_message_id = await fetch_and_print_new_agent_response(
                thread_id=thread.id,
                agents_client=agents_client,
                last_message_id=last_message_id,
            )
            print(f"Run status: {run.status}")

        print(f"Run finished with status: {run.status}, ID: {run.id}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # List the messages, print them and save the result in research_summary.md file.
        all_messages = []
        async for message in agents_client.messages.list(thread_id=thread.id, order="asc"):
            all_messages.append(message)

        if all_messages:
            print_messages_and_save_summary(all_messages)

        # Clean-up and delete the agent and thread once the run is finished.
        # NOTE: Comment out this line if you plan to reuse the agent later.
        await agents_client.threads.delete(thread.id)
        print("Deleted thread")
        await agents_client.delete_agent(agent.id)
        print("Deleted agent")


if __name__ == "__main__":
    asyncio.run(main()) 