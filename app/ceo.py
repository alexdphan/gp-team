# Import os for environment variable handling
import os
# Import deque for task list management
from collections import deque
# Import typing for type hinting
from typing import Dict, List, Optional, Any

# Import dotenv for loading environment variables from .env file
from dotenv import load_dotenv
# Import pydantic for data validation and BaseModel class
from pydantic import BaseModel, Field

# Import langchain classes and functions
from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma

# Import chromadb errors for handling exceptions
from chromadb import errors as chromadb_errors

# Load environment variables from .env file
load_dotenv()

# Set up and assert environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
assert SERPAPI_API_KEY, "SERPAPI_API_KEY environment variable is missing from .env"

YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"
table_name = YOUR_TABLE_NAME

OBJECTIVE = os.getenv("OBJECTIVE", "Write a weather report for SF today")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv(
    "FIRST_TASK", "Develop a task list"))
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

# Define the embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Set up the Vector Store
persist_directory = "db"
vectorstore = Chroma(table_name, embeddings_model,
                     persist_directory=persist_directory)
vectorstore.persist()

# Define the Chains
class RoleCreationChain(LLMChain):
    """Chain to create roles."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an AI tasked with creating roles for a team based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["user_id", "chroma_instance", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class TaskCreationAssignChain(LLMChain):
    """Chain to generates tasks."""

    tools: List[Tool] = []

    @classmethod
    def from_llm(cls, llm: BaseLLM, tools: List[Tool] = [], verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_assignment_template = (
            "You are an AI tasked with creating and assigning tasks to a team based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            " Create a task list for each team member and assign tasks according to their expertise."
        )
        prompt = PromptTemplate(
            template=task_assignment_template,
            input_variables=[
                "user_id",
                "chroma_instance",
                "objective",
                "team_members_expertise",
            ],
        )
        return cls(prompt=prompt, llm=llm, tools=tools, verbose=verbose)

    def run(self, *args, **kwargs):
        # Use the Search tool to gather more information about the objective
        search_tool = next((tool for tool in self.tools if tool.name == "Search"), None)
        if search_tool:
            search_results = search_tool.func(query=kwargs["objective"])
            # Add search results to the input variables
            kwargs["search_results"] = search_results

        # Use the TodoChain tool to generate a to-do list based on the objective
        todo_tool = next((tool for tool in self.tools if tool.name == "TodoChain"), None)
        if todo_tool:
            todo_list = todo_tool.func(objective=kwargs["objective"])
            # Add the generated to-do list to the input variables
            kwargs["todo_list"] = todo_list

        return super().run(*args, **kwargs)

class ReportCreationChain(LLMChain):
    """Chain to generate a report."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an AI tasked with creating a report for a team based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["user_id", "chroma_instance", "objective", "team_members_expertise"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ReviseCreationChain(LLMChain):
    """Chain to revise the outputs."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an AI tasked with revising the outputs of each TeamMember based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["user_id", "chroma_instance", "objective", "team_members_expertise"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    

# Assigning todo_prompt prompt that takes in an objective (prompt) and returns a todo list (response)
# this is todo_chain will be used as a tool to generate a todo list
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
# todo_chain is an instance of the LLMChain class
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
# search is an instance of the SerpAPIWrapper class
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
# uses the tools that include search and todo_chain to come up with a todo list
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TodoChain",
        func=todo_chain.run,
        description="useful for generating a to-do list based on a given objective",
    ),
]

# A prompt that takes in an objective, a task, and a context and returns an answer in zero-shot, outputs in the form of a string
prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
) # not being used atm


# Define the CEO Controller #
def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    # response is a string, so we split it into a list of strings
    new_tasks = response.split("\n")
    # return a list of dictionaries, each dictionary has a task_name key value being the task name
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    # task_names is a list of strings, each string is a task name
    task_names = [t["task_name"] for t in task_list]
    # next_task_id is an integer, it is the next task id
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    # prioritized_task_list is a list of dictionaries, each dictionary has a task_id key value being the task id and a task_name key value being the task name
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        # task_parts is a list of strings, the first string is the task id and the second string is the task name
        task_parts = task_string.strip().split(".", 1)
        # if the length of task_parts is 2, then the task id is the first string and the task name is the second string
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append(
                {"task_id": task_id, "task_name": task_name})
    # return the list of dictionaries
    return prioritized_task_list


def _get_top_tasks(vectorstore: Chroma, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    try:
        results = vectorstore.similarity_search_with_score(query=query, k=k)
    except chromadb_errors.NoIndexException:
        return []

    # results is a list of tuples, each tuple has a vectorstore item and a score
    if not results:
        return []

    # The rest of the function remains the same
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))

    tasks = []
    for item in sorted_results:
        try:
            tasks.append(str(item.metadata["task"]))
        except KeyError:
            print(f"")

    return tasks

def execute_task(
    vectorstore: Chroma,
    execution_chain: LLMChain,
    objective: str,
    task_info: Dict[str, Any],
    k: int = 5,
) -> str:
    """Execute a task."""
    # while true, get top k tasks, if not enough, reduce k by 1, if k == 0, break. break doesn't give a value, so context is an empty list
    while True:
        try:
            context = _get_top_tasks(
                vectorstore=vectorstore, query=objective, k=k)
            break
        except chromadb_errors.NotEnoughElementsException:
            k -= 1
            if k == 0:
                context = []
                break
    # Execute the task
    result = execution_chain.run(
        objective=objective, context=context, task=task_info["task_name"]
    )
    # store the result on the vectorstore
    result_id = f"result_{task_info['task_id']}"
    vectorstore.add_texts(
        texts=[result],
        metadatas=[
            {"task": task_info["task_name"]}
        ],  # Set 'task' key in metadata here, using task_info
        ids=[result_id],
    )
    return result

# Define the CEO Class
class CEO(BaseModel):
    """Controller model for the CEO's TeamMember agent."""

    task_list: deque = Field(default_factory=deque)  # list of tasks
    # chain generating new tasks
    role_creation_chain: RoleCreationChain = Field(...)
    task_creation_assign_chain: TaskCreationAssignChain = Field(...)
    report_creation_chain: ReportCreationChain = Field(...)
    revise_creation_chain: ReviseCreationChain = Field(...)
    task_id_counter: int = Field(1)  # counter for task ids
    # vectorstore for storing results
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None  # maximum number of iterations

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def run(self, objective: str, chroma_instance: Any, user_id: str):
        iteration = 0
        team_members_expertise = []

        while True:
            # Step 1: Create roles for team members
            if not team_members_expertise:
                team_members_expertise = self.role_creation_chain.run(
                    user_id=user_id, chroma_instance=chroma_instance, objective=objective
                )

            # Step 2: Create and assign tasks to team members
            task_lists = self.task_creation_assign_chain.run(
                user_id=user_id,
                chroma_instance=chroma_instance,
                objective=objective,
                team_members_expertise=team_members_expertise,
            )

            # * Get outputs from team members here (starting in the second loop)

            # Step 3: Create a report
            report = self.report_creation_chain.run(
                user_id=user_id,
                chroma_instance=chroma_instance,
                objective=objective,
                team_members_expertise=team_members_expertise,
            )

            # Step 4: Revise and give constructive critique and feedback
            feedback = self.revise_creation_chain.run(
                user_id=user_id,
                chroma_instance=chroma_instance,
                objective=objective,
                team_members_expertise=team_members_expertise,
            )

            # Print the report and feedback
            print("Report:\n", report)
            print("Feedback:\n", feedback)

            # Ask for additional feedback from the Board of Directors (the user)
            user_input = input(
                "Enter 'approve' to approve the report cycle, 'stop' to stop the process, or press Enter to run continuously: "
            )

            if user_input.lower() == "approve":
                break
            elif user_input.lower() == "stop":
                return

            iteration += 1
            if self.max_iterations is not None and iteration == self.max_iterations:
                break
            
    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, tools: List[Tool] = [], verbose: bool = False, **kwargs
    ) -> "CEO":
        """Initialize the CEO Controller."""
        role_creation_chain = RoleCreationChain.from_llm(llm, verbose=verbose)
        task_creation_assign_chain = TaskCreationAssignChain.from_llm(llm, tools=tools, verbose=verbose)
        report_creation_chain = ReportCreationChain.from_llm(llm, verbose=verbose)
        revise_creation_chain = ReviseCreationChain.from_llm(llm, verbose=verbose)

        return cls(
            role_creation_chain=role_creation_chain,
            task_creation_assign_chain=task_creation_assign_chain,
            report_creation_chain=report_creation_chain,
            revise_creation_chain=revise_creation_chain,
            vectorstore=vectorstore,
            **kwargs,
        )

# Instantiate the CEO class using the from_llm method
llm = OpenAI(temperature=0)
verbose = False
max_iterations: Optional[int] = 3
ceo = CEO.from_llm(
    llm=llm, vectorstore=vectorstore, tools=tools, verbose=verbose, max_iterations=max_iterations
)

# Run the CEO with the given objective, chroma_instance, and user_id
ceo.run(objective=OBJECTIVE, chroma_instance="ChromaDB_instance", user_id="user_1")









