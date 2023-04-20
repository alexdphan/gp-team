import os
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
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

def main():
    ceo = CEO(
        chroma_instance=chroma_instance,
        role_creation_chain=role_creation_chain,
        task_creation_assign_chain=task_creation_assign_chain,
        report_creation_chain=report_creation_chain,
        revise_creation_chain=revise_creation_chain,
    )

    user_message_handler = UserMessageHandler(ceo)

    print("Welcome to the CEO AI!")
    print("You can type your questions or commands, and the CEO AI will respond accordingly.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        else:
            response = user_message_handler.process_message(user_input)
            print("CEO AI:", response)

if __name__ == "__main__":
    main()