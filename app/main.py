import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from ceo import CEO, RoleCreationChain, TaskCreationAssignChain, ReportCreationChain, ReviseCreationChain, UserMessageHandler
from fastapi.middleware.cors import CORSMiddleware
from ceo import CEO, RoleCreationChain, TaskCreationAssignChain, ReportCreationChain, ReviseCreationChain, UserMessageHandler

# Set Variables
load_dotenv()

# Read API keys from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"
table_name = YOUR_TABLE_NAME

persist_directory = "db"

chroma_instance = (
    Chroma(table_name, embeddings_model,
           persist_directory=persist_directory)
)

# Initialize LLM instance with the API key
llm = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the CEO module and all chains
role_creation_chain = RoleCreationChain.from_llm(llm)
task_creation_assign_chain = TaskCreationAssignChain.from_llm(llm)
report_creation_chain = ReportCreationChain.from_llm(llm)
revise_creation_chain = ReviseCreationChain.from_llm(llm)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputModel(BaseModel):
    user_input: Optional[str] = None
    objective: Optional[str] = None
    num_team_members: Optional[int] = None
    feedback: Optional[str] = None

async def get_ceo() -> CEO:
    # Your existing CEO initialization code
    ceo = CEO(
        chroma_instance=chroma_instance,
        role_creation_chain=role_creation_chain,
        task_creation_assign_chain=task_creation_assign_chain,
        report_creation_chain=report_creation_chain,
        revise_creation_chain=revise_creation_chain,
    )
    return ceo

@app.post("/process")
async def process_inputs(input_data: InputModel, ceo: CEO = Depends(get_ceo)):
    user_input = input_data.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Invalid input")

    response = await ceo.run_workflow(user_input)
    return {"response": response}
