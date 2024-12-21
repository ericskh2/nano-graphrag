from dataclasses import dataclass,field
from mistralai import Mistral

@dataclass
class RetrievalStrategyAgent:
    """
    Determines the retrieval strategy (global/local/naive) for graph-rag
    """
    llm_base_url: str = "llm_base_url_here"
    llm_api_key: str = "WkR7n3wrHUFz02P7NgjweucocW2yIRFZ"
    llm_model_name: str = "mistral-large-latest"

    system_prompt: str = field(
        default="""
            You are an intelligent agent responsible for determining the appropriate query type based on the provided scenario. Your query types include local, global, and naive queries. I will give you a scenario and you need to analyze the scenario and choose the best query type to obtain the required information efficiently. Please give the answer only.
        """,
        init=False,
    )

    def __post_init__(self):
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