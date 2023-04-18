
from langchain import LLMChain, PromptTemplate
import openai
from .team_member import TeamMember
from .utils import OPENAI_API_KEY, search_tool
from langchain.vectorstores import Chroma
from langchain.llms import BaseLLM
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain, OpenAI, PromptTemplate
import uuid
from typing import List
from langchain.agents import Tool
from pydantic import UUID4, BaseModel, Field
from chromadb.errors import NoIndexException

class CustomLLMChainWrapper(BaseModel):
    user_id: int = Field(...)
    llm_chain: LLMChain = Field(...)

    def __init__(self, user_id, prompt, llm=None):
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)
        super().__init__(user_id=user_id, llm_chain=self.llm_chain)

    def __getattr__(self, name):
        llm_chain = object.__getattribute__(self, "llm_chain")
        if hasattr(llm_chain, name):
            return getattr(llm_chain, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class TaskCreationAssignChain(CustomLLMChainWrapper):
    def __init__(self, user_id, llm=None):
        task_assignment_template = (
            "You are an AI tasked with creating and assigning tasks to a team based on their expertise and the team's objective."
            " Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Create a task list for each team member and assign tasks according to their expertise."
        )
        task_creation_assign_prompt = PromptTemplate(
            template=task_assignment_template,
            input_variables=["objective", "team_members_expertise"],
        )
        super().__init__(user_id=user_id, prompt=task_creation_assign_prompt, llm=llm)

    @classmethod
    def from_llm(cls, llm):
        return cls(llm=llm)


class ReportCreationChain(CustomLLMChainWrapper):
    def __init__(self, user_id, llm=None):
        report_creation_template = (
            "You are an AI tasked with creating a report of what the CEO did, the work each TeamMember has done, and the problems the team is facing."
            " Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Use the information provided to create a detailed report on the team's progress, challenges, and accomplishments."
        )
        report_creation_prompt = PromptTemplate(
            template=report_creation_template,
            input_variables=["objective", "team_members_expertise"],
        )
        super().__init__(user_id=user_id, prompt=report_creation_prompt, llm=llm)

    @classmethod
    def from_llm(cls, llm):
        return cls(llm=llm)


class ReviseCreationChain(CustomLLMChainWrapper):
    def __init__(self, user_id, llm=None):
        revise_creation_template = (
            "You are an AI tasked with revising and giving constructive critique and feedback to each TeamMember based on their outputs, the initial task list, and the Chromdb database instance."
            " Objective: {objective}."
            " Team members and their expertise roles: {team_members_expertise}."
            " Analyze the team members' work and provide specific feedback and revisions for each team member to improve their performance."
        )
        revise_creation_prompt = PromptTemplate(
            template=revise_creation_template,
            input_variables=["objective", "team_members_expertise"],
        )
        super().__init__(user_id=user_id, prompt=revise_creation_prompt, llm=llm)

    @classmethod
    def from_llm(cls, llm):
        return cls(llm=llm)


class CEO:
    def __init__(self, objective=None, user_id=None, team_members=None, default_llm=None, default_chroma=None, fast_llm_model=None, smart_llm_model=None, api_key=OPENAI_API_KEY):
        self.role = "CEO"
        self.objective = objective if objective else input("Enter the team's objective: ")
        self.user_id = user_id if user_id else uuid.uuid4()
        self.chroma = default_chroma
        self.api_key = api_key

        self.fast_llm_model = fast_llm_model
        self.smart_llm_model = smart_llm_model

        self.task_creation_assign_chain = TaskCreationAssignChain(self.user_id, default_llm) if default_llm else TaskCreationAssignChain(self.user_id)
        self.task_creation_assign_chain = TaskCreationAssignChain(self.user_id, default_llm) if default_llm else TaskCreationAssignChain(self.user_id)
        self.report_creation_chain = ReportCreationChain(self.user_id, default_llm) if default_llm else ReportCreationChain(self.user_id)
        self.revise_creation_chain = ReviseCreationChain(self.user_id, default_llm) if default_llm else ReviseCreationChain(self.user_id)

        self.team_members = team_members if team_members is not None else self.create_team_members()

        self.executor = AgentExecutor()

    def generate_expertise_keywords(self, prompt: str, chroma: Chroma, api_key: str, model_key: str = "text-davinci-002") -> List[str]:
        expertise_prompt_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Generate a list of expertise keywords related to the objective."
        )
        expertise_prompt = expertise_prompt_template.format(objective=prompt)

        openai.api_key = api_key

        response = openai.Completion.create(
            engine=model_key,
            prompt=expertise_prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )

        generated_text = response.choices[0].text.strip()
        expertise_keywords = generated_text.split(', ')

        # Check if the Chroma instance has an index, and create one if it doesn't
        try:
            results = chroma.similarity_search(query="query", k=int)
        except NoIndexException:
            return []

        
        # Filter the keywords using the Chroma instance
        expertise_keywords = [keyword for keyword in expertise_keywords if chroma.similarity_search(keyword, k=1)]
        return expertise_keywords

    def create_team_members(self):
        generated_keywords = self.generate_expertise_keywords(prompt=self.objective, chroma=self.chroma, api_key=self.api_key)
        expertise_keywords = [keyword for keyword in generated_keywords if keyword in [result[0] for result in self.chroma.similarity_search(keyword, k=1)]]
        team_members = [TeamMember(role=expertise) for expertise in expertise_keywords]
        return team_members

    def create_and_assign_tasks(self):
        task_creation_tool = Tool(
            name="task_creation",
            function=self.task_creation_assign_chain,
            input_variables=["objective", "team_members_expertise"],
            output_variables=["assigned_task_lists"],
        )

        team_members_expertise = [member.role for member in self.team_members]

        task_creation_result = self.executor.execute(
            tool=(task_creation_tool, search_tool),
            input_data={"objective": self.objective, "team_members_expertise": team_members_expertise},
        )

        assigned_task_lists = task_creation_result["assigned_task_lists"]

        return assigned_task_lists

    def create_report(self):
        report_creation_tool = Tool(
            name="report_creation",
            function=self.report_creation_chain,
            input_variables=["objective", "team_members_expertise"],
            output_variables=["report"],
        )

        team_members_expertise = [member.role for member in self.team_members]

        report_creation_result = self.executor.execute(
            tool=(report_creation_tool, search_tool),
            input_data={"objective": self.objective, "team_members_expertise": team_members_expertise},
        )

        report = report_creation_result["report"]

        return report

    def revise_and_give_feedback(self):
        revise_creation_tool = Tool(
            name="revise_creation",
            function=self.revise_creation_chain,
            input_variables=["objective", "team_members_expertise"],
            output_variables=["revisions_and_feedback"],
        )

        team_members_expertise = [member.role for member in self.team_members]

        revisions_output = self.executor.execute(
            tool=(revise_creation_tool, search_tool),
            input_data={"objective": self.objective, "team_members_expertise": team_members_expertise},
        )

        revisions_and_feedback = revisions_output["revisions_and_feedback"]

        return revisions_and_feedback

    def set_task_creation_assign_chain(self, llm):
        self.task_creation_assign_chain = TaskCreationAssignChain.from_llm(llm)

    def set_report_creation_chain(self, llm):
        self.report_creation_chain = ReportCreationChain.from_llm(llm)

    def set_revise_creation_chain(self, llm):
        self.revise_creation_chain = ReviseCreationChain.from_llm(llm)