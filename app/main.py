from typing import Optional
import uuid
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.ceo import CEO
from app.utils import search_tool, OPENAI_API_KEY, SERP_API_KEY, Tool
from app.team_member import TaskData, TeamMember
from langchain import LLMChain, OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from app.config import config # import loaded config from config.py

app = FastAPI()

class TaskData(BaseModel):
    objective: str

# Initialize team members
team_members = [
    TeamMember(role="Data Scientist"),
    TeamMember(role="Software Engineer"),
    TeamMember(role="Product Manager"),
]

openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# using the fast_llm_model for now from config.py
openai_llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_kwargs={"model": config.fast_llm_model})
# setting the main chroma instance to the one from config.py, which is a chroma instance
main_chroma_instance = Chroma(config.chroma_name, openai_embeddings)

ceo = CEO(objective="Search top restaurants in nyc", default_chroma=main_chroma_instance, default_llm=openai_llm, api_key=OPENAI_API_KEY)

@app.get("/")
def read_root(input_prompt: Optional[str] = None):
    """Read root route."""
    if input_prompt:
        return {"Input Prompt": input_prompt}
    else:
        return {"Error": "No input prompt provided"}

@app.post("/submit_objective")
async def submit_objective(task_data: TaskData, user_id: Optional[str] = None):
    """Submit objective route."""
    objective_id = uuid.uuid4()
    objective = task_data.objective
    
    # Add some texts to the Chroma instance to create an index
    main_chroma_instance.add_texts(["dummy text"])

    # Delete the dummy collection
    main_chroma_instance.delete_collection()

    # Now you can call the generate_expertise_keywords method
    ceo = CEO(objective="Search top restaurants in nyc", default_chroma=main_chroma_instance, default_llm=openai_llm, api_key=OPENAI_API_KEY)
        
    # Call the relevant methods to create and assign tasks, create a report, and revise and give feedback
    assigned_task_lists = ceo.create_and_assign_tasks()
    report = ceo.create_report()
    revisions_and_feedback = ceo.revise_and_give_feedback()

    # Save the objective with the generated UUID
    # You can replace this with your own logic to store the objective
    objectives = {}
    objectives[objective_id] = objective

    return {
            "Objective ID": str(objective_id),
            "Objective": objective,
            "Assigned Task Lists": assigned_task_lists,
            "Report": report,
            "Revisions and Feedback": revisions_and_feedback,
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)