"""Utility module for chroma and prompt template initialization."""

from langchain.vectorstores import Chroma
from langchain import GoogleSearchAPIWrapper, OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from config import config # Import the loaded config module from config.py
import uuid

# Load environment variables from the .env file
load_dotenv()

# Get the value of the OPENAI_API_KEY environment variable
OPENAI_API_KEY = config.openai_api_key
# Get the value of the SERP_API_KEY environment variable
SERP_API_KEY = config.serp_api_key
# Get the value of the GOOGLE_API_KEY environment variable
GOOGLE_API_KEY = config.google_api_key

# Initialize ChromaDB with embeddings and persist_directory
persist_directory = 'db'
# Pass the API key as an argument
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define which of the more general tools (from langchain) the agent can use to answer user queries

# Serp from Langchain
search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)
search_tool = Tool(
    name="Search",
    func=search.run,
    description="A search tool utilizing SerpAPI to provide answers to questions related to current events, news, and general information."
)

# Google from Langchain
# google = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY)
# google_tool = Tool(
#     name="Google",
#     func=google.run,
#     description="A search tool utilizing Google Search API to provide answers to questions related to current events, news, and general information."
# )