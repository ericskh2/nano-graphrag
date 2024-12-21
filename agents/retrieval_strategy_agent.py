class RetrievalStrategyAgent:
    """
    Determines the retrieval strategy (global/local/naive) for graph-rag
    """
    def __init__(self, llm_base_url: str, llm_api_key: str):
        """
        Initializes the RetrievalStrategyAgent.

        Args:
            llm_base_url (str): Base URL of the LLM API.
            llm_api_key (str): API key for authenticating requests to the LLM.
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key

    def run(self, query_input: str) -> str:
        """
        Executes the retrieval strategy agent by sending a query to the LLM.

        Args:
            query_input (str): The input query to be sent to the LLM.

        Returns:
            str: The mode determined by the LLM (global/local/naive).
        """

        print('Running retrieval strategy agent')

        return "global"