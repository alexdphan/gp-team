import abc
import os
import openai
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# Singleton is a metaclass that ensures only one instance of a class is created,
# providing a global point of access to that instance. It is used for managing
# shared resources or configurations throughout an application.
class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]

# AbstractSingleton is an abstract base class that is also a singleton. It allows
# creating classes that follow a common interface (abstract base class) and have
# only one instance (singleton) throughout the application.
class AbstractSingleton(abc.ABC, metaclass=Singleton):
    pass # for now?

class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        """Initialize the Config class"""
        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False

        self.chroma_name = os.getenv("CHROMA_NAME", "default_chroma_name")
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serp_api_key = os.getenv("SERPAPI_API_KEY")
        self.temperature = float(os.getenv("TEMPERATURE", "1"))
        self.use_azure = os.getenv("USE_AZURE") == 'True'
        self.execute_local_commands = os.getenv('EXECUTE_LOCAL_COMMANDS', 'False') == 'True'

        # if self.use_azure:
        #     self.load_azure_config()
        #     openai.api_type = self.openai_api_type
        #     openai.api_base = self.openai_api_base
        #     openai.api_version = self.openai_api_version

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

        # User agent headers to use when browsing web
        # Some websites might just completely deny request with an error code if no user agent was found.
        self.user_agent_header = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
        # self.redis_host = os.getenv("REDIS_HOST", "localhost")
        # self.redis_port = os.getenv("REDIS_PORT", "6379")
        # self.redis_password = os.getenv("REDIS_PASSWORD", "")
        # self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == 'True'
        # self.memory_index = os.getenv("MEMORY_INDEX", 'auto-gpt')
        # Note that indexes must be created on db 0 in redis, this is not configurable.

        self.memory_backend = os.getenv("MEMORY_BACKEND", 'local')
        # Initialize the OpenAI API client
        openai.api_key = self.openai_api_key

    def set_continuous_mode(self, value: bool):
        """Set the continuous mode value."""
        self.continuous_mode = value

    def set_continuous_limit(self, value: int):
        """Set the continuous limit value."""
        self.continuous_limit = value

    def set_fast_llm_model(self, value: str):
        """Set the fast LLM model value."""
        self.fast_llm_model = value

    def set_smart_llm_model(self, value: str):
        """Set the smart LLM model value."""
        self.smart_llm_model = value

    def set_fast_token_limit(self, value: int):
        """Set the fast token limit value."""
        self.fast_token_limit = value

    def set_smart_token_limit(self, value: int):
        """Set the smart token limit value."""
        self.smart_token_limit = value

    def set_openai_api_key(self, value: str):
        """Set the OpenAI API key value."""
        self.openai_api_key = os.env("OPENAI_API_KEY", value)

    def set_google_api_key(self, value: str):
        """Set the Google API key value."""
        self.google_api_key = value

    def set_debug_mode(self, value: bool):
        """Set the debug mode value."""
        self.debug_mode = value
        
config = Config()

# use file later for organization