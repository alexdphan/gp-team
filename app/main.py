import os
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from app.ceo import (
    CEO,
    RoleCreationChain,
    TaskCreationAssignChain,
    ReportCreationChain,
    ReviseCreationChain,
)
from colorama import Fore, Style

def main():

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

    # Initialize the LLM, Chroma, and chain instances
    llm = OpenAI(api_key=OPENAI_API_KEY)
    chroma_instance = (
        Chroma(table_name, embeddings_model,
               persist_directory=persist_directory)
    )  # Pass the API key when initializing the Chroma instance
    role_creation_chain = RoleCreationChain.from_llm(llm)
    task_creation_assign_chain = TaskCreationAssignChain.from_llm(llm)
    report_creation_chain = ReportCreationChain.from_llm(llm)
    revise_creation_chain = ReviseCreationChain.from_llm(llm)

    # Initialize the CEO instance
    ceo = CEO(
            chroma_instance=chroma_instance,
            role_creation_chain=role_creation_chain,
            task_creation_assign_chain=task_creation_assign_chain,
            report_creation_chain=report_creation_chain,
            revise_creation_chain=revise_creation_chain,
        )
    # Set the objective
    objective = "Develop a new AI product"

    # Set the number of loops or use a stopping condition
    number_of_loops = 5

    print(Fore.GREEN + "1. RoleCreationChain:" + Style.RESET_ALL)
    ceo.create_team_members(objective=objective, num_team_members=3)

    for i in range(number_of_loops):
        
        user_id = ceo.get_new_user_id()

        print(Fore.GREEN + "2. TaskCreationAssignChain:" + Style.RESET_ALL)
        ceo.assign_tasks_to_team_members(objective=objective)
        print(Fore.GREEN + "2. Assigned tasks to team members:" + Style.RESET_ALL)

        print(Fore.GREEN + "4. ReportCreationChain:" + Style.RESET_ALL)
        report = ceo.report_creation_chain.run(
            user_id=user_id,
            objective=objective,
            chroma_instance=ceo.chroma_instance,
            team_members_expertise=ceo.get_team_members_expertise(),
        )
        
        print(Fore.GREEN + "3. Getting outputs from team members:" + Style.RESET_ALL)
        for team_member_id, team_member_instance in ceo.team_members.items():
            ceo.receive_output(team_member_id, team_member_instance.execute_tasks(chroma_instance=ceo.chroma_instance))
        
        print(Fore.GREEN + "3. Received outputs from team members:" + Style.RESET_ALL)


        print(Fore.GREEN + "5. ReviseCreationChain:" + Style.RESET_ALL)
        revised_team_outputs = ceo.revise_creation_chain({
                'user_id': user_id,
                'objective': objective,
                'chroma_instance': ceo.chroma_instance,
                'team_members_expertise': ceo.get_team_members_expertise(),
                'user_feedback': feedback if i > 0 else None  # Pass user_feedback to ReviseCreationChain
            })

        print(f"Report (Cycle {i+1}):")
        print(report.strip().replace("\\n", "\n"))
        print(f"\nRevised Team Outputs (Cycle {i+1}):") 
        revised_outputs = revised_team_outputs.get('revised_outputs', '')
        print(revised_outputs.strip().replace("\\n", "\n"))

        # Get input from the user (Board of Directors)
        user_input = input("Enter 'approve' to approve the report cycle or provide your feedback to continue: ")

        feedback = None
        if user_input.lower() != "approve":
            feedback = user_input

if __name__ == "__main__":
    main()

