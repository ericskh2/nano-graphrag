from dataclasses import dataclass, field
from mistralai import Mistral
from openai import AsyncOpenAI
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

@dataclass
class FeedbackAgentMistral:
    """
    A class representing an agent for providing feedback based on the query and initial response.
    """
    llm_base_url: str = None
    llm_api_key: str = None
    llm_model_name: str = None

    embedding_model_name: str = None
    embedding_model_dim: int = None
    embedding_model_max_tokens: int = None

    system_prompt: str = field(
        default="""
                    You are an intelligent assistant responsible for giving feedback for the given generated response. You will get a query input and a current response of this query input about the retrieved contents or inserted corpus. Please give the feedback with advantages and disadvantages respectively. Ensure the output is in JSON format.
                """,
        init=False,
    )
    """
    Attributes:
        llm_base_url (str): The base URL for the LLM API.
        llm_api_key (str): The API key for authenticating with the LLM API.
    """

    def __init__(self, llm_base_url: str, llm_api_key: str, llm_model_name: str = "mistral-large-latest", embedding_model_name: str = "nomic-embed-text:ctx32k", embedding_model_dim: int = 768, embedding_model_max_tokens: int = 32000):
        """
        Initializes the FeedbackAgent with the provided LLM base URL and API key.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_model_dim = embedding_model_dim
        self.embedding_model_max_tokens = embedding_model_max_tokens

        # embedding_model_name: str = "nomic-embed-text"
        # embedding_model_dim: int = 768
        # embedding_model_max_tokens: int = 8192

        # llm_base_url: str = "llm_base_url_here"
        # llm_api_key: str = "WkR7n3wrHUFz02P7NgjweucocW2yIRFZ"
        # llm_model_name: str = "mistral-large-latest"
        pass

    async def llm_model_if_cache(
            self, prompt, history_messages=[], **kwargs
    ) -> str:
        client = Mistral(api_key=self.llm_api_key)
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        # Get the cached response if having-------------------
        # hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        # if hashing_kv is not None:
        #     args_hash = compute_args_hash(self.llm_model_name, messages)
        #     if_cache_return = await hashing_kv.get_by_id(args_hash)
        #     if if_cache_return is not None:
        #         return if_cache_return["return"]
        # -----------------------------------------------------

        settings = {'response_format': {'type': 'json_object'}}
        response = client.chat.complete(
            model=self.llm_model_name, messages=messages, **settings
        )

        # Cache the response if having-------------------
        # if hashing_kv is not None:
        #     await hashing_kv.upsert(
        #         {args_hash: {"return": response.choices[0].message.content, "model": self.llm_model_name}}
        #     )
        # -----------------------------------------------------
        return response.choices[0].message.content

    def embedding_model(self):
        # @wrap_embedding_func_with_attrs(
        #     embedding_dim=self.embedding_model_dim,
        #     max_token_size=self.embedding_model_max_tokens,
        # )
        async def ollama_embedding(texts: list[str]) -> np.ndarray:
            embed_text = []
            for text in texts:
                data = ollama.embeddings(model=self.embedding_model_name, prompt=text)
                embed_text.append(data["embedding"])

            return embed_text

        return ollama_embedding


    def run(self, work_directory_path: str, query_input: str, initial_query_response: str, ) -> str:
        """
        Generates feedback for the given query and initial response.

        Args:
            work_directory_path (str): The directory path related to the feedback process.
            query_input (str): The input query.
            initial_query_response (str): The initial response to the query.

        Returns:
            str: Feedback for the initial response.
        """
        ollama_embedding = self.embedding_model()
        ollama_embedding.embedding_dim=self.embedding_model_dim
        ollama_embedding.max_token_size=self.embedding_model_max_tokens
        rag = GraphRAG(
            working_dir=work_directory_path,
            best_model_func=self.llm_model_if_cache,
            cheap_model_func=self.llm_model_if_cache,
            embedding_func=ollama_embedding
        )
        query_question = f"""
            This is the query input: "{query_input}".
            This is the current response: "{initial_query_response}".
            Please give the feedback with advantages and disadvantages respectively.
        """

        print('Running feedback agent')
        # Simulate feedback generation
        feedback_response = rag.query(
            query=query_question,
            param=QueryParam(mode="global")
        )
        return feedback_response
    
@dataclass
class FeedbackAgent:
    """
    A class representing an agent for providing feedback based on the query and initial response.
    """
    llm_base_url: str = None
    llm_api_key: str = None
    llm_model_name: str = None

    embedding_model_name: str = None
    embedding_model_dim: int = None
    embedding_model_max_tokens: int = None

    system_prompt: str = field(
        default="""
                    You are an intelligent assistant responsible for giving feedback for the given generated response. You will get a query input and a current response of this query input about the retrieved contents or inserted corpus. Please give the feedback with advantages and disadvantages respectively.
                """,
        init=False,
    )
    """
    Attributes:
        llm_base_url (str): The base URL for the LLM API.
        llm_api_key (str): The API key for authenticating with the LLM API.
    """

    def __init__(self, llm_base_url: str, llm_api_key: str, llm_model_name: str, embedding_model_name: str = "nomic-embed-text:ctx32k", embedding_model_dim: int = 768, embedding_model_max_tokens: int = 32000):
        """
        Initializes the FeedbackAgent with the provided LLM base URL and API key.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_model_dim = embedding_model_dim
        self.embedding_model_max_tokens = embedding_model_max_tokens
        pass

    async def llm_model_if_cache(
            self, prompt, history_messages=[], **kwargs
    ) -> str:
        client = AsyncOpenAI(
            api_key=self.llm_api_key, base_url=self.llm_base_url
        )
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        # Get the cached response if having-------------------
        # hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        # if hashing_kv is not None:
        #     args_hash = compute_args_hash(self.llm_model_name, messages)
        #     if_cache_return = await hashing_kv.get_by_id(args_hash)
        #     if if_cache_return is not None:
        #         return if_cache_return["return"]
        # -----------------------------------------------------

        response = await client.chat.completions.create(
            model=self.llm_model_name, messages=messages, **kwargs
        )

        # Cache the response if having-------------------
        # if hashing_kv is not None:
        #     await hashing_kv.upsert(
        #         {args_hash: {"return": response.choices[0].message.content, "model": self.llm_model_name}}
        #     )
        # -----------------------------------------------------
        return response.choices[0].message.content

    def embedding_model(self):
        # @wrap_embedding_func_with_attrs(
        #     embedding_dim=self.embedding_model_dim,
        #     max_token_size=self.embedding_model_max_tokens,
        # )
        async def ollama_embedding(texts: list[str]) -> np.ndarray:
            embed_text = []
            for text in texts:
                data = ollama.embeddings(model=self.embedding_model_name, prompt=text)
                embed_text.append(data["embedding"])

            return embed_text

        return ollama_embedding


    def run(self, work_directory_path: str, query_input: str, initial_query_response: str, ) -> str:
        """
        Generates feedback for the given query and initial response.

        Args:
            work_directory_path (str): The directory path related to the feedback process.
            query_input (str): The input query.
            initial_query_response (str): The initial response to the query.

        Returns:
            str: Feedback for the initial response.
        """
        ollama_embedding = self.embedding_model()
        ollama_embedding.embedding_dim=self.embedding_model_dim
        ollama_embedding.max_token_size=self.embedding_model_max_tokens
        rag = GraphRAG(
            working_dir=work_directory_path,
            best_model_func=self.llm_model_if_cache,
            cheap_model_func=self.llm_model_if_cache,
            embedding_func=ollama_embedding
        )
        query_question = f"""
            This is the query input: "{query_input}".
            This is the current response: "{initial_query_response}".
            Please give the feedback with advantages and disadvantages respectively.
        """

        print('Running feedback agent')
        # Simulate feedback generation
        feedback_response = rag.query(
            query=query_question,
            param=QueryParam(mode="global")
        )
        return feedback_response
