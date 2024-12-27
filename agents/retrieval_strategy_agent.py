from dataclasses import dataclass,field
from mistralai import Mistral
from openai import AsyncOpenAI

@dataclass
class RetrievalStrategyAgentMistral:
    """
    Determines the retrieval strategy (global/local/naive) for graph-rag
    """
    llm_base_url: str = None
    llm_api_key: str = None
    llm_model_name: str = None

    system_prompt: str = field(
        default="""
            You are an intelligent agent responsible for determining the appropriate query type based on the provided scenario. Your query types include local, global, and naive queries. I will give you a scenario and you need to analyze the scenario and choose the best query type to obtain the required information efficiently. Please give the answer only.
        """,
        init=False,
    )

    def __init__(self, llm_base_url: str, llm_api_key: str, llm_model_name: str = "mistral-large-latest"):
        """
        Initializes the RetrievalStrategyAgent with the provided LLM base URL and API key.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        pass


    """mode: Literal["local", "global", "naive"]"""
    def run(self, query_input: str) -> str:
        """
        Executes the retrieval strategy agent by sending a query to the LLM.

        Args:
            query_input (str): The input query to be sent to the LLM.

        Returns:
            str: The mode determined by the LLM (global/local/naive).
        """
        client = Mistral(api_key=self.llm_api_key)
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query_input})
        chat_response = client.chat.complete(
            model=self.llm_model_name,
            messages=messages
        )
        print('Running retrieval strategy agent')
        result: str = chat_response.choices[0].message.content
        return result.lower()
    
@dataclass
class RetrievalStrategyAgent:
    """
    Determines the retrieval strategy (global/local/naive) for graph-rag
    """
    llm_base_url: str = None
    llm_api_key: str = None
    llm_model_name: str = None

    system_prompt: str = field(
        default="""
            You are an intelligent agent responsible for determining the appropriate query type based on the provided scenario. Your query types include local, global, and naive queries. I will give you a scenario and you need to analyze the scenario and choose the best query type to obtain the required information efficiently. Please give the answer only.
        """,
        init=False,
    )

    def __init__(self, llm_base_url: str, llm_api_key: str, llm_model_name: str):
        """
        Initializes the RetrievalStrategyAgent with the provided LLM base URL and API key.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        pass


    """mode: Literal["local", "global", "naive"]"""
    async def run(self, query_input: str) -> str:
        """
        Executes the retrieval strategy agent by sending a query to the LLM.

        Args:
            query_input (str): The input query to be sent to the LLM.

        Returns:
            str: The mode determined by the LLM (global/local/naive).
        """
        client = AsyncOpenAI(
            api_key=self.llm_api_key, base_url=self.llm_base_url
        )
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query_input})
        chat_response = await client.chat.completions.create(
            model=self.llm_model_name, messages=messages
        )
        print('Running retrieval strategy agent')
        result: str = chat_response.choices[0].message.content
        return result.lower()