# models.py
"""Module containing GPTeamMember and CEO classes."""

from pydantic import BaseModel

class TaskData(BaseModel):
    user_id: int
    task: str
    objective: str

class TeamMember:
    """Class representing team members with GPT roles."""

    def __init__(self, role):
        self.role = role
        self.feedback = None  # Add a feedback attribute

    def process_task(self, task, prompt_templates, chroma_vector_store):
        """Process a given task."""
        # Implement your task processing logic here
        # e.g., use Langchain and Chroma to process task and return the result

        return "Processed task result"  # Placeholder result