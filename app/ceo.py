
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
    TaskPrioritizationChain,
    ExecutionChain,
)

class RoleCreationChain(LLMChain):
    """Chain to create roles."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        prompt_template = (
            "You are an AI tasked with creating roles and their expertise for a team based on the team's objective."
            " User ID: {user_id}. Objective: {objective}. Number of Team Members: {num_team_members}."
            "\n\nPlease create exactly {num_team_members} roles for the team, and provide your response in the following format:"
            "\nRole 1: Role Name, Expertise 1, Expertise 2, ..."
            "\nRole 2: Role Name, Expertise 1, Expertise 2, ..."
            "\n..."
            "\nRole {num_team_members}: Role Name, Expertise 1, Expertise 2, ..."
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_id", "objective", "num_team_members"],  
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
                "objective",
                "team_members_expertise",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


    def generate_new_task_list(self, vectorstore: Chroma, result: Dict, task_description: str, task_list: List[str], objective: str, top_k: int) -> List[Dict]:

        """
        Get the next task based on team members' expertise, the objective, and the Chroma instance. We use it then to get the top k tasks from the Chroma instance as a starting point and add them to the current task list.

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
        # get top k tasks from the chain instance from the TaskCreationAssignChain class (method below)
        top_tasks = self.get_top_tasks(vectorstore, task_description, top_k) 
         # add the top k tasks to the current task list
        task_list.extend(top_tasks)
         # join the tasks in the task list into a string
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
            # return a list of tasks
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
            " Please update the report with the latest information."
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
            template=prompt_template,
            input_variables=["user_id", "objective", "team_members_expertise"],
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
            " If there are no outputs to revise, provide only what you have done so far."
            " The user provided the following feedback: {user_feedback}. With this, please revise the outputs accordingly."
            "\n\nAction: {{action}}"
            " Action Input: {{action_input}}"
            " Thought: {{thought}}"
            " Final Answer: {{final_answer}}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_id", "objective", "team_members_expertise", "user_feedback"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def _response_parser(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Custom response parser to extract the revised outputs as a dictionary."""
        revised_output = response["choices"][0]["text"]
        revised_output = revised_output.strip().split("\n")

        revised_output_dict = {}
        for line in revised_output:
            key, value = line.split(": ", 1)
            revised_output_dict[key.strip()] = value.strip()

        return revised_output_dict


# this class is for parsing lists of lists, e.g. [[1,2,3],[4,5,6]]
class ListParser: # outside of CEO class because it is not specific to CEO
    @staticmethod
    def parse(input_str: str, delimiter: str = ";", item_separator: str = ","):
        result = []
        groups = input_str.split(delimiter)
        for group in groups:
            result.append(group.split(item_separator))
        return result
        # for example, if input_str = "1,2,3;4,5,6", then result = [[1,2,3],[4,5,6]]

class CEO:
    """ CEO class for the CEO role in the team. The CEO is responsible for creating a team, assigning tasks to team members, creating a report, and revising the outputs of each team member. """
    def __init__(
        self,
        llm: BaseLLM,
        role_creation_chain: LLMChain,
        task_creation_assign_chain: LLMChain,
        report_creation_chain: LLMChain,
        revise_creation_chain: LLMChain,
        # task_prioritization_chain: TaskPrioritizationChain, # from team_member.py
        # execution_chain: ExecutionChain,
    ):
        self.llm = llm
        self.role_creation_chain = role_creation_chain
        self.task_creation_assign_chain = task_creation_assign_chain
        self.report_creation_chain = report_creation_chain
        self.revise_creation_chain = revise_creation_chain
        self.team_members: Dict[str, TeamMember] = {}
        self.team_member_outputs = {}
        self.user_id = 0
        self.user_feedback = ""
        
        # added from team_member.py
        # needs to add .from_llm() to the end of each chain because it is a class method in each chain class, we get this from team_member.py
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm)
        self.execution_chain = ExecutionChain.from_llm(llm)

        # generic_prompt = PromptTemplate(
        #     # variable_name is placeholder for the variable name
        #     template="{variable_name}",  # Specify the variable name inside the placeholder
        #     input_variables=["variable_name"],  # Add the variable name to the input_variables list
        # )
        
        # self.task_prioritization_chain: LLMChain
        # self.execution_chain = LLMChain
      
    def get_new_user_id(self):
        """Get a new user id for a new team member. For example, 1, 2, etc. This is used to create a new team member."""
        self.user_id += 1
        return self.user_id


    def create_team_members(self, objective: str, num_team_members: int):
        roles_input = {
            "user_id": self.user_id,
            "objective": objective,
            "num_team_members": num_team_members,
            "team_members_expertise": self.get_team_members_expertise(),
        }
        roles = self.role_creation_chain.run(roles_input)
        
        # Parse the roles output into a list of strings
        expertise_roles = [role.strip() for role in roles.split("\n") if role.strip()]
        expertise_roles = expertise_roles[:num_team_members]  # Keep only the first num_team_members roles

        for role in expertise_roles:
            team_member_id = self.get_new_user_id()
            team_member = create_team_member(
                name={"team_member_id": team_member_id, "role": role},
                user_id=team_member_id,
                expertise_role=role,
                task_list=[],
                task_prioritization_chain=self.task_prioritization_chain,
                execution_chain=self.execution_chain,
                objective=objective,
            )
            self.team_members[team_member_id] = team_member

        team_members_str = "\n".join(
            [f"Created team member {team_member_id} with expertise role: {role.title()}" for team_member_id, role in zip(self.team_members, expertise_roles)]
        )

        return self.team_members


        # print(team_members_str)



        # print(f"Created team members with roles and assignments based on their expertise:\n{roles}")
        expertise_roles = roles.strip().split("\n")[:num_team_members]
        # print(f"Expertise roles: {expertise_roles}")

        team_members_str = "\n".join(
            [f"Created team member {team_member_id} with expertise role: {role.title()}" for team_member_id, role in zip(self.team_members, expertise_roles)]
        )
        print(team_members_str)

        return self.team_members
    
    def assign_tasks_to_team_members(self, objective: str):
        """Creates and assign tasks to each team member based on their expertise and the team's objective. This is for the CEO to assign tasks to each team member. This is done by running the task creation and assignment chain."""
        team_members_expertise = self.get_team_members_expertise()

        # run the task creation and assignment chain, creating a task list for each team member
        task_lists = self.task_creation_assign_chain.run(
            user_id=self.user_id,
            objective=objective,
            team_members_expertise=team_members_expertise,
            num_team_members=len(self.team_members),  
        )
        # parse the task lists into a list of lists, for example, [[1,2,3],[4,5,6]]
        task_lists = ListParser.parse(task_lists)
        
        # For each team member, assign the task list to the team member
        for user_id, task_list in zip(self.team_members.keys(), task_lists):
            self.team_members[user_id].task_list = task_list
            # print(f"For user {user_id}, assigned task list: {task_list}")


    def get_team_members_expertise(self):
        """Get the expertise of each team member. This is used to assign tasks to each team member in the assign_tasks_to_team_members function."""
        
        # print(f"Team member {self.user_id} has expertise roles: {self.team_members.items()}")
        return {user_id: team_member.expertise_role for user_id, team_member in self.team_members.items()}

    def receive_output(self, team_member_id, output):
        """Receive output from a team member. This is used to receive output from a team member."""
        self.team_member_outputs[team_member_id] = output

    def handle_feedback(self, feedback: str):
        """Handle feedback from the user. This is used to handle feedback from the user."""
        self.user_feedback = feedback  # Set the user_feedback attribute on the CEO object
        
    def collect_team_member_outputs(self):
        """Collect the outputs from each team member. This is used to collect the outputs from each team member."""
        print(f"Collecting outputs from team member {self.user_id}...")
        for team_member_id, team_member in self.team_members.items():
            self.team_member_outputs[team_member_id] = team_member.execute_tasks()
            
    def generate_reports(self):
        """Generate reports for the CEO. This is used to generate reports for the CEO."""
        report = self.report_creation_chain.run(
            user_id=self.user_id,
            team_member_outputs=self.team_member_outputs,
            team_members=self.team_members,
            user_feedback=self.user_feedback,
        )
        print(f"report: {report}")
        return report


    #=========== Workflow ===========#
    # the functions would call the chains in the order of the workflow
    def run_workflow(self, objective: str, num_team_members: int, user_feedback: str = None):
        # Input validation for objective and num_team_members
        if not objective or not isinstance(num_team_members, int) or num_team_members <= 0:
            raise ValueError("Invalid input for objective or num_team_members.")

        try:
            # RoleCreationChain
            print("\n===== Executing RoleCreationChain =====")
            print("Creating team members...")
            self.create_team_members(objective=objective, num_team_members=num_team_members)
            
            # Create a formatted string for printing the team members
            team_members_str = "\n".join(
                [f"Team Member ID: {team_member_id}, Role: {team_member.expertise_role}"
                for team_member_id, team_member in self.team_members.items()]
            )
            print(f"Team members:\n{team_members_str}")
            print("===== Finished RoleCreationChain =====\n")


            # looped steps
            satisfied = False
            while not satisfied:
                
                # TaskCreationChain
                print("\n===== Executing TaskCreationAssignChain =====")
                print("Assigning tasks to team members...")
                self.assign_tasks_to_team_members(objective=objective)
                print("===== Finished TaskCreationAssignChain =====\n")
                
                # Initialize a dictionary to store team member outputs
                team_outputs = {}

                # For each team member, prioritize tasks and execute tasks
                for user_id, team_member in self.team_members.items():
                    print(f"Prioritizing tasks for user {user_id}...")
                    prioritized_task_list = team_member.prioritize_tasks()
                    print(f"Prioritized task list for user {user_id}: {prioritized_task_list}")
                    team_member.task_list = prioritized_task_list
                    print(f"Executing tasks for user {user_id}...")
                    team_member_outputs = team_member.execute_tasks()
                    print(f"Outputs for user {user_id}: {team_member_outputs}")

                # Generate report using the report creation chain (executed by the CEO)
                print("Creating a report based on the progress we've made so far...")
                report = self.report_creation_chain.run(
                    user_id=self.user_id,
                    objective=objective,
                    team_members_expertise=self.get_team_members_expertise(),
                    team_outputs=team_outputs,
                )
                print(f"Created a report: {report}")

                # Get user feedback or check if the output meets the desired condition
                user_feedback = input("Are you satisfied with the results? (yes/no): ")
                satisfied = user_feedback.lower() == "yes"

            print("Workflow complete!")
            
            # Return a dictionary of the report and team member outputs
            return {
                "report": report,
                "revised_team_outputs": team_outputs,
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

        # RoleCreationChain
        # TaskCreationAssignChain (LOOPED)
        # ReportCreationChain (LOOPED)
        # ReviseCreationChain (LOOPED)
        # TaskPrioritizationChain (LOOPED)
        # ExecutionChain (LOOPED)

class UserMessageHandler:
    """This class is used to handle user messages. This includes setting the objective, providing feedback, and processing user messages."""
    def __init__(self, ceo: CEO):
        """Initialize the UserMessageHandler class."""
        self.ceo = ceo
        
        # Initialize the conversation memory and the LLM chain
        # Uses ConversationBufferWindowMemory and PromptTemplate from the LLM package to be able to store the conversation history
        self.conversation_memory = ConversationBufferWindowMemory()
        # Uses basic prompt template to be able to store the user message
        generic_prompt = PromptTemplate(
            template="User message: {user_message}",
            input_variables=["user_message"],
        )
        self.llm_chain = LLMChain(llm=self.ceo.role_creation_chain.llm, prompt=generic_prompt, memory=self.conversation_memory)
        
    def process_message(self, message: str, objective: str = None, num_team_members: int = None):
        """This is used to process a user message. This includes taking in a user message, the objective, and the number of team members. Then it returns a response."""
        message = message.lower().strip()

        # If the user message contains "set objective", then the objective and number of team members are requested from the user
        if "set objective" in message:
            # Input handling can be modified for non-interactive contexts
            objective = input("Enter the objective: ").strip()
            num_team_members = int(input("Enter the number of team members: ").strip())
            
            report, revised_team_outputs = self.ceo.run_workflow(objective, num_team_members)
            
            # print("\nReport:")
            # print(report)
            # print("\nRevised Team Outputs:")
            # print(revised_team_outputs)
            return report, revised_team_outputs

        # If the user types "provide feedback", then the user is prompted to enter their feedback and the feedback is handled
        elif "provide feedback" in message:
            # Input handling can be modified for non-interactive contexts
            feedback = input("Enter your feedback: ").strip()
            self.ceo.handle_feedback(feedback)
            return "Feedback received."

        # Additional user message handling cases can be added here
        # Else, the user message is processed by the LLM chain as a regular user message, which returns a natural language response
        else:
            response = self.llm_chain.run(user_message=message)
            return response

    # Changed to regular method (def) instead of async def. This is because the user message handler is not used in an async context. For example, the user message handler is not used in a fastapi endpoint.
    def handle_input(self, user_input: str) -> str:
        response = self.process_message(user_input)
        return response
    