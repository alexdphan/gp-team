"""Main module for FastAPI application and its route definitions."""

from fastapi import FastAPI
from .ceo import CEO
from .models import GPTeamMember
from .utils import initialize_prompt_templates, init_chroma_store

app = FastAPI()

ceo = CEO()
team_members = [
    GPTeamMember(role="NLP Specialist"),
    GPTeamMember(role="Computer Vision Specialist"),
    GPTeamMember(role="Data Analysis Specialist"),
]

chroma_vector_store = init_chroma_store("vector_store/")
prompt_templates = initialize_prompt_templates()


@app.get("/")
def read_root():
    """Read root route."""
    return {"Hello": "World"}


@app.post("/submit_task")
async def submit_task(task: str):
    """Submit task route."""
    assigned_member = ceo.assign_task(task, team_members)
    processed_task = assigned_member.process_task(
        task, prompt_templates, chroma_vector_store
    )
    return {"result": processed_task}