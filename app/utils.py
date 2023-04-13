# utils.py
"""Utility module for chroma and prompt template initialization."""

from langchain.vectorstores import Chroma

def initialize_prompt_templates():
    """Initialize prompt templates."""
    prompt_templates = {
        'nlp_summarization':
            'Please summarize the following text:\n\n{content}\n\n',
        # Add more custom prompt templates as needed
        # for different roles and tasks
    }
    return prompt_templates

def init_chroma_store(path):
    """Initialize chroma store."""
    try:
        store = Chroma()
    except FileNotFoundError:
        store = Chroma.from_documents([], save_path=path)

    return store