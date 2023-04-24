
from collections import deque
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
from chromadb import errors as chromadb_errors
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferWindowMemory

from team_member import (
    TeamMember,
    create_team_member,
    # TaskPrioritizationChain,
    # ExecutionChain,
)

# Define the Chains
class RoleCreationChain(LLMChain):
    """Chain to create roles."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        prompt_template = (
            "You are an AI tasked with creating roles for a team based on their expertise and the team's objective. Format: Role Name, Expertise Role 1, Expertise Role 2, ..."
            " User ID: {user_id}. Objective: {objective}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            "\n\nPlease provide your response in the following format:"
            "\nAction: [describe the action]"
            "\nAction Input: [describe the input for the action]"
            "\nThought: [describe the thought process]"
            "\nFinal Answer: [provide the final answer]"
            "\n\nExample:"
            "\nAction: Create roles"
            "\nAction Input: Objective, Chroma Instance"
            "\nThought: Based on the objective and the chroma instance, the necessary roles can be..."
            "\nFinal Answer: Role Name, Expertise Role 1, Expertise Role 2, ..."
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_id", "chroma_instance", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose) 
       

class TaskCreationAssignChain(LLMChain):
    """Chain to generate and assign tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_assignment_template = (
            "You are an AI tasked with creating and assigning tasks to a team based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            " Create a task list for each team member and assign tasks according to their expertise."
            "\n\nPlease provide your response in the following format:"
            "\nAction: [describe the action]"
            "\nAction Input: [describe the input for the action]"
            "\nThought: [describe the thought process]"
            "\nFinal Answer: [provide the final answer]"
            "\n\nExample:"
            "\nAction: Generate and assign tasks"
            "\nAction Input: Objective, Chroma Instance, Team members and their expertise roles"
            "\nThought: Based on the objective, chroma instance, and team members' expertise, I can create tasks that align with their skills..."
            "\nFinal Answer: Task list for each team member"
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
        return cls(prompt=prompt, llm=llm, verbose=verbose)

        
    def generate_new_task_list(self, vectorstore: Chroma, result: Dict, task_description: str, task_list: List[str], objective: str, top_k: int) -> List[Dict]:
        """
        Get the next task based on team members' expertise, the objective, and the Chroma instance.

        Args:
            vectorstore (Chroma): A Chroma instance containing information to consider.
            result (Dict): A dictionary containing relevant information about the current state.
            task_description (str): A description of the task to generate.
            task_list (List[str]): A list of existing tasks.
            objective (str): The team's objective.
            top_k (int): The number of top tasks to consider.

        Returns:
            List[Dict]: A list of dictionaries, each containing a task name.
        """
        top_tasks = self.get_top_tasks(vectorstore, task_description, top_k)
        task_list.extend(top_tasks)
        incomplete_tasks = ", ".join(task_list)

        response = self.run(
            result=result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
            objective=objective,
        )
        # response is a string, so we split it into a list of strings
        new_tasks = response.split(",")  # Use a more specific delimiter
        # return a list of dictionaries, each dictionary has a task_name key value being the task name
        return [{"task_name": task_name.strip()} for task_name in new_tasks if task_name.strip()]

    
    # used to help the chain generate tasks for each team member based on their expertise, the objective, and the Chroma instance
    # used within the generate_new_task_list function
    def get_top_tasks(self, vectorstore: Chroma, query: str, k: int) -> List[str]:
            """
            Get the top k tasks based on the query. This is used when the CEO is creating a new task list,
            getting the top k tasks from the Chroma instance as a starting point. Important because the CEO
            is not an expert in the field, so the CEO needs to rely on the Chroma instance to get the top k tasks.
            """
            try:
                results = vectorstore.similarity_search_with_score(query=query, k=k)
            except chromadb_errors.NoIndexException:
                return []
            if not results:
                return []

            sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))

            tasks = []
            for item in sorted_results:
                try:
                    tasks.append(str(item.metadata["task"]))
                except KeyError:
                    # Log a relevant message or remove the logging statement if it's not necessary
                    logging.warning(f"KeyError occurred while processing the metadata for item: {item}")

            return tasks

class ReportCreationChain(LLMChain):
    """Chain to generate a report."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        prompt_template = (
            "You are an AI tasked with creating a report for a team based on their expertise and the team's objective."
            " User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            " Please update the report with the latest information."
            "\n\nAction: {{action}}"
            " Action Input: {{action_input}}"
            " Thought: {{thought}}"
            " Final Answer: {{final_answer}}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_id", "chroma_instance", "objective", "team_members_expertise"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ReviseCreationChain(LLMChain):
    """Chain to revise the outputs of each team member based on their expertise and the team's objective."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:                    
        """Get the response parser."""
        prompt_template = (
            "You are an AI tasked with revising the outputs of each TeamMember based on their expertise and the team's objective." 
            "User ID: {user_id}. Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Consider the information from the Chroma Instance: {chroma_instance}."
            " If there are no outputs to revise, provide only what the you have done so far."
            " The user provided the following feedback: {user_feedback}. With this, please revise the outputs accordingly."
            "\n\nAction: {{action}}"
            " Action Input: {{action_input}}"
            " Thought: {{thought}}"
            " Final Answer: {{final_answer}}"
            )
        prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["user_id", "chroma_instance", "objective", "team_members_expertise", "user_feedback"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
class ListParser: # outside of CEO class because it is not specific to CEO
    @staticmethod
    def parse(input_str: str, delimiter: str = ";", item_separator: str = ","):
        result = []
        groups = input_str.split(delimiter)
        for group in groups:
            result.append(group.split(item_separator))
        return result

class CEO:
    def __init__(
        self,
        chroma_instance: Chroma,
        role_creation_chain: LLMChain,
        task_creation_assign_chain: LLMChain,
        report_creation_chain: LLMChain,
        revise_creation_chain: LLMChain,
    ):
        self.chroma_instance = chroma_instance
        self.role_creation_chain = role_creation_chain
        self.task_creation_assign_chain = task_creation_assign_chain
        self.report_creation_chain = report_creation_chain
        self.revise_creation_chain = revise_creation_chain
        self.team_members: Dict[str, TeamMember] = {}
        self.team_member_outputs = {}
        self.user_id = 0
        self.user_feedback = ""

        self.task_prioritization_chain = LLMChain.from_llm(role_creation_chain.llm)
        self.execution_chain = LLMChain.from_llm(revise_creation_chain.llm)

    def get_new_user_id(self):
        self.user_id += 1
        return f"user_{self.user_id}"

    def create_team_members(self, objective, num_team_members):
        roles = self.role_creation_chain.run(
            user_id=self.user_id,
            objective=objective,
            chroma_instance=self.chroma_instance,
        )
        expertise_roles = roles.strip().split("\n")[:num_team_members]

        for i, role in enumerate(expertise_roles):
            team_member_id = f"{self.user_id}-{i + 1}"
            team_member = create_team_member(
                user_id=team_member_id,
                expertise_role=role,
                task_list=[],
                task_prioritization_chain=self.task_prioritization_chain,
                execution_chain=self.execution_chain,
                objective=objective,
            ) 
            self.team_members[team_member_id] = team_member
    

    def assign_tasks_to_team_members(self, objective: str):
        team_members_expertise = self.get_team_members_expertise()

        task_lists = self.task_creation_assign_chain.run(
            user_id=self.user_id,
            objective=objective,
            chroma_instance=self.chroma_instance,
            team_members_expertise=team_members_expertise,
        )
        task_lists = ListParser.parse(task_lists)

        for user_id, task_list in zip(self.team_members.keys(), task_lists):
            self.team_members[user_id].task_list = task_list


    def get_team_members_expertise(self):
        return {user_id: team_member.expertise_role for user_id, team_member in self.team_members.items()}
        # reutrns a dictionary of user_id: expertise_role for each team member, this is generated from the role creation chain

    def execute_chains(self, objective: str, num_team_members: int, user_feedback: str):
        self.create_team_members(objective=objective, num_team_members=num_team_members)
        self.assign_tasks_to_team_members(objective=objective)

        team_outputs = {}
        for user_id, team_member in self.team_members.items():
            prioritized_task_list = team_member.prioritize_tasks(chroma_instance=self.chroma_instance)
            team_member.task_list = prioritized_task_list
            team_outputs[user_id] = team_member.execute_tasks(chroma_instance=self.chroma_instance)

        report = self.report_creation_chain.run(
            user_id=self.user_id,
            objective=objective,
            chroma_instance=self.chroma_instance,
            team_members_expertise=self.get_team_members_expertise(),
        )

        self.team_member_outputs = team_outputs

        revised_team_outputs = self.revise_creation_chain.run(
            user_id=self.user_id,
            objective=objective,
            chroma_instance=self.chroma_instance,
            team_members_expertise=self.get_team_members_expertise(),
            user_feedback=self.user_feedback,
            team_outputs=team_outputs,
        )

        return {
            "report": report,
            "revised_team_outputs": revised_team_outputs,
            "team_member_outputs": team_outputs,
        }

    def receive_output(self, team_member_id, output):
        self.team_member_outputs[team_member_id] = output

    def get_team_members_expertise(self):
        team_members_expertise = {}
        for team_member_id, team_member_instance in self.team_members.items():
            team_members_expertise[team_member_id] = team_member_instance.expertise_role
        return team_members_expertise

    def handle_feedback(self, feedback: str):
        self.user_feedback = feedback  # Set the user_feedback attribute on the CEO object
        
    def collect_team_member_outputs(self):
        for team_member_id, team_member in self.team_members.items():
            self.team_member_outputs[team_member_id] = team_member.execute_tasks()
            
    def generate_reports(self):
        report = self.report_creation_chain.run(
            user_id=self.user_id,
            team_member_outputs=self.team_member_outputs,
            team_members=self.team_members,
            chroma_instance=self.chroma_instance,
            user_feedback=self.user_feedback,
        )
        return report


    ########## Workflow ##########
    def run_workflow(self, objective: str, num_team_members):
        print("Creating team members...")
        self.create_team_members(objective=objective, num_team_members=num_team_members)
        print("Assigning tasks to team members...")
        self.assign_tasks_to_team_members(objective=objective)
        print("Executing chains...")
        results = self.execute_chains(objective=objective, num_team_members=num_team_members, user_feedback=self.user_feedback)
        report = results["report"]
        revised_team_outputs = results["revised_team_outputs"]
        print("Workflow complete!")

        return report, revised_team_outputs


class UserMessageHandler:
    def __init__(self, ceo: CEO):
        self.ceo = ceo
        
        self.conversation_memory = ConversationBufferWindowMemory()
        generic_prompt = PromptTemplate(
            template="User message: {user_message}",
            input_variables=["user_message"],
        )
        self.llm_chain = LLMChain(llm=self.ceo.role_creation_chain.llm, prompt=generic_prompt, memory=self.conversation_memory)
        
    def process_message(self, message: str, objective: str = None, num_team_members: int = None):
        message = message.lower().strip()

        if "set objective" in message:
            if not objective:
                objective = input("Enter the objective: ").strip()
            if not num_team_members:
                num_team_members = int(input("Enter the number of team members: ").strip())
            report, revised_team_outputs = self.ceo.run_workflow(objective, num_team_members)
            print("\nReport:")
            print(report)
            print("\nRevised Team Outputs:")
            print(revised_team_outputs)
            return report, revised_team_outputs

        elif "provide feedback" in message:
            feedback = input("Enter your feedback: ").strip()
            self.ceo.handle_feedback(feedback)
            return "Feedback received."

        else:
            response = self.llm_chain.run(user_message=message)
            return response

    async def handle_input(self, user_input: str) -> str:
        response = self.process_message(user_input)
        return response