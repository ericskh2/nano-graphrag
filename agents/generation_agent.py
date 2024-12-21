from dataclasses import dataclass, field
from mistralai import Mistral
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

@dataclass
class GenerationAgent:
    """
    A class representing an agent for text generation tasks using an LLM.
    """
    llm_base_url: str = "llm_base_url_here"
    llm_api_key: str = "WkR7n3wrHUFz02P7NgjweucocW2yIRFZ"
    llm_model_name: str = "mistral-large-latest"

    embedding_model_name: str = "nomic-embed-text"
    embedding_model_dim: int = 768
    embedding_model_max_tokens: int = 8192

    system_prompt: str = field(
        default="""
                You are an intelligent assistant responsible for generating a response based on the retrieved contents, inserted corpus and the query input.
            """,
        init=False,
    )
    """
    Attributes:
        llm_base_url (str): The base URL for the LLM API.
        llm_api_key (str): The API key for authenticating with the LLM API.
    """
    
    def __post_init__(self):
        pass

    async def llm_model_if_cache(
        self, prompt, history_messages=[], **kwargs
    ) -> str:
        client = Mistral(api_key=self.llm_api_key)
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        # Get the cached response if having-------------------
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            args_hash = compute_args_hash(self.llm_model_name, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        # -----------------------------------------------------

        response = client.chat.complete(
            model=self.llm_model_name, messages=messages, **kwargs
        )

        # Cache the response if having-------------------
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": response.choices[0].message.content, "model": self.llm_model_name}}
            )
        # -----------------------------------------------------
        return response.choices[0].message.content

    def embedding_model(self) -> callable:
        @wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_model_dim,
            max_token_size=self.embedding_model_max_tokens,
        )
        async def ollama_embedding(texts: list[str]) -> np.ndarray:
            embed_text = []
            for text in texts:
                data = ollama.embeddings(model=self.embedding_model_name, prompt=text)
                embed_text.append(data["embedding"])

            return embed_text

        return ollama_embedding


    def run(self, work_directory_path: str, retrieval_strategy: str, query_input: str) -> str:
        """
        Executes the agent to generate a response based on the input query.

        Args:
            work_directory_path (str): The directory path related to the query.
            retrieval_strategy (str): The strategy used for data retrieval (e.g., global/local/naive).
            query_input (str): The input string for text generation.

        Returns:
            str: The generated response.
        """
        rag = GraphRAG(
            working_dir=work_directory_path,
            best_model_func=self.llm_model_if_cache,
            cheap_model_func=self.llm_model_if_cache,
            embedding_func=self.embedding_model
        )

        print('Running generation agent')
        generated_response = rag.query(
            query=query_input,
            param=QueryParam(mode=retrieval_strategy)
        )
        
        return generated_response