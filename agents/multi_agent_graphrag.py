from typing import List
import argparse

# local agents folder
from query_agent import run_query_agent  # Import the function

class RAGQueryResponse:
    def __init__(self, query: str, response: str, score: float, iteration_cnt: int):
        """
        Represents a response from a Retrieval-Augmented Generation (RAG) query.

        :param query: The original query string.
        :param response: The generated response to the query.
        :param score: Confidence score for the response.
        """

        self.query = query
        self.response = response
        self.score = score
        self.iteration_cnt = iteration_cnt

class OrchestratorAgentResponse:
    def __init__(self, query_input: str, response_list: List[RAGQueryResponse], total_iteration_cnt: int):
        self.query_input = query_input
        self.response_list = response_list
        self.total_iteration_cnt = total_iteration_cnt
    
    def get_best_response(self) -> RAGQueryResponse:
        """
        Return the RAGQueryResponse with the highest score.
        If multiple responses have the same score, return the later one.
        """
        best_response = None
        for response in self.response_list:
            if best_response is None or response.score > best_response.score or (
                response.score == best_response.score and response.iteration_cnt > best_response.iteration_cnt):
                best_response = response
        return best_response
    
    def get_total_iteration_cnt(self) -> int:
        """
        Return the total number of iterations performed.
        
        :return: The total iteration count.
        """
        return self.total_iteration_cnt

def run_orchestrator_agent(work_directory_path: str, query_input_path: str, query_output_path: str, score_threshold: float = 4, iteration_threshold: int = 3) -> OrchestratorAgentResponse:
    """
    Run the orchestrator agent. Loop until the score is greater than or equals to the threshold or the iteration count exceeds the limit.

    :param query_input_path: The path to the input query txt to be processed.
    :param score_threshold: The threshold for the score to stop iterating.
    :param iteration_threshold: The maximum number of iterations allowed.

    :return: OrchestratorAgentResponse containing the results.
    """

    # Open the query txt file and read its contents into a string
    with open(query_input_path, 'r') as file:
        query_input_str = file.read()

    max_score = None
    iteration_cnt = 0

    while (max_score is None or max_score < score_threshold) and iteration_cnt < iteration_threshold:
        query_response: str = run_query_agent(work_directory_path, query_input_str)  # Query the agent for a response
        iteration_cnt += 1
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that accepts command-line arguments.")

    parser.add_argument(
        '--work_directory', 
        type=str,
        required=True,
        help='The path to the work directory.'
    )

    parser.add_argument(
        '--query_input_path', 
        type=str,
        required=True,
        help='The input path to the query txt file.'
    )

    parser.add_argument(
        '--query_output_path', 
        type=str,
        required=True,
        help='The output path to the answer file.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    work_directory_path = args.work_directory
    query_input_path = args.query_input_path
    query_output_path = args.query_output_path

    run_orchestrator_agent(work_directory_path=work_directory_path, query_input_path=query_input_path, query_output_path=query_output_path, score_threshold=4, iteration_threshold=3)