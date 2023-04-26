from collections import deque
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
from chromadb import errors as chromadb_errors

# Define the Chains for TeamMembers

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks for a TeamMember."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an AI tasked with prioritizing tasks for a TeamMember based on their expertise and the team's objective."
            " User ID: {user_id}. Expertise Role: {expertise_role}."
            " Task List: {task_list}."
            " Objective: {objective}. Overall Feedback: {overall_feedback}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            "\n\nPlease provide your response in the following format:"
            "\nAction: [describe the action]"
            "\nAction Input: [describe the input for the action]"
            "\nThought: [describe the thought process]"
            "\nFinal Answer: [list prioritized tasks]"
            "\n\nExample:"
            "\nAction: Prioritize tasks"
            "\nAction Input: Expertise Role, Task List, Objective, Overall Feedback, Chroma Instance"
            "\nThought: Based on the expertise role and the team's objective, the tasks can be prioritized as..."
            "\nFinal Answer: Task 1, Task 2, Task 3, ..."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["user_id", "chroma_instance", "expertise_role", "task_list", "objective", "overall_feedback"],
    )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ExecutionChain(LLMChain):
    """Chain to execute tasks for a TeamMember."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_execution_template = (
            "You are an AI tasked with executing tasks for a TeamMember based on their expertise and the team's objective."
            " User ID: {user_id}. Expertise Role: {expertise_role}."
            " Task List: {task_list}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            "\n\nPlease provide your response in the following format:"
            "\nAction: [describe the action]"
            "\nAction Input: [describe the input for the action]"
            "\nThought: [describe the thought process]"
            "\nFinal Answer: [list task execution results]"
            "\n\nExample:"
            "\nAction: Execute tasks"
            "\nAction Input: Expertise Role, Task List, Chroma Instance"
            "\nThought: Based on the expertise role, I can execute the tasks as follows..."
            "\nFinal Answer: Result 1, Result 2, Result 3, ..."
        )
        prompt = PromptTemplate(
            template=task_execution_template,
            input_variables=["user_id", "chroma_instance", "expertise_role", "task_list"],
    )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# Define the TeamMember class

class TeamMember:
    def __init__(
        self,
        llm: BaseLLM,
        user_id: str,
        expertise_role: str,
        task_list: List[str],
        task_prioritization_chain: LLMChain,
        execution_chain: LLMChain,
        objective: str,
        overall_feedback: str,
        chroma_instance: Chroma,

    ):
        self.llm = llm
        self.user_id = user_id
        self.expertise_role = expertise_role
        self.task_list = task_list
        self.task_prioritization_chain = task_prioritization_chain
        self.execution_chain = execution_chain
        self.objective = objective
        self.overall_feedback = overall_feedback
        self.chroma_instance = chroma_instance




    def prioritize_tasks(self):
        """Prioritize tasks using the TaskPrioritizationChain."""
        
        return self.task_prioritization_chain.run(
            user_id=self.user_id,
            chroma_instance=self.chroma_instance,
            expertise_role=self.expertise_role,
            task_list=self.task_list,
            objective=self.objective,
            overall_feedback=self.overall_feedback,
        )

    def execute_tasks(self):
        """Execute tasks using the ExecutionChain."""
        
        return self.execution_chain.run(
            user_id=self.user_id,
            chroma_instance=self.chroma_instance,
            expertise_role=self.expertise_role,
            task_list=self.task_list,
            objective=self.objective,
            overall_feedback=self.overall_feedback,
        )

# Additional functionality for the TeamMember class can be added here as needed.
# Example of creating TeamMember objects with the chains
def create_team_member(
    user_id: str,
    expertise_role: str,
    task_list: List[str],
    task_prioritization_chain: TaskPrioritizationChain,
    execution_chain: ExecutionChain,
    objective: str,
    overall_feedback: Optional[str] = None,
    chroma_instance: Chroma = None,
) -> TeamMember:
    print(f"Creating team member with the role: {expertise_role}")
    return TeamMember(
        llm=LLMChain,
        user_id=user_id,
        expertise_role=expertise_role,
        task_list=task_list,
        task_prioritization_chain=task_prioritization_chain,
        execution_chain=execution_chain,
        objective=objective,
        overall_feedback=overall_feedback,
        chroma_instance=chroma_instance,
    )

