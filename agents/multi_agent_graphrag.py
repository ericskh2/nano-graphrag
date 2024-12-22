from typing import List
import argparse
import os

# local agents folder
from retrieval_strategy_agent import RetrievalStrategyAgent, RetrievalStrategyAgentMistral
from generation_agent import GenerationAgent, GenerationAgentMistral
from feedback_agent import FeedbackAgent, FeedbackAgentMistral
from refinement_agent import RefinementAgent, RefinementAgentMistral
from scoring_agent import ScoringAgent, ScoringAgentMistral

class RAGQueryResponse:
    def __init__(self, query: str, response: str, score: float, iteration_cnt: int):
        """
        Represents a single response from a Retrieval-Augmented Generation (RAG) query.

        :param query: The original query string.
        :param response: The generated response to the query.
        :param score: Confidence score for the response.
        """

        self.query = query
        self.response = response
        self.score = score
        self.iteration_cnt = iteration_cnt

    def get_score(self):
        """
        Return the score of the response

        :return: score of response
        """

        return self.score
    
class OrchestratorAgentResponse:
    """
    An object containing all responses from a Retrieval-Augmented Generation (RAG) query.
    """

    def __init__(self, query_input: str):
        self.query_input = query_input
        self.response_list = []
        self.total_iteration_cnt = 0
    
    def add_response(self, response: RAGQueryResponse):
        self.response_list.append(response)

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
        Return the total number of iterations performed. (length of the response list)
        
        :return: The total iteration count.
        """

        return len(self.response_list)

def run_orchestrator_agent(work_directory_path: str, query_input_path: str, query_output_path: str, score_threshold: float = 4, iteration_threshold: int = 3, mistral: bool = False) -> OrchestratorAgentResponse:
    """
    Run the orchestrator agent. Loop until the score is greater than or equals to the threshold or the iteration count exceeds the limit.

    :param query_input_path: The path to the input query txt to be processed.
    :param score_threshold: The threshold for the score to stop iterating.
    :param iteration_threshold: The maximum number of iterations allowed.

    :return: OrchestratorAgentResponse containing the results.
    """

    try:
        llm_base_url = os.getenv("LLM_BASE_URL")
    except:
        raise Exception("LLM_BASE_URL environment variable is not set.")

    try:
        llm_api_key = os.getenv("LLM_API_KEY")
    except:
        raise Exception("LLM_API_KEY environment variable is not set.")
    
    try:
        llm_model = os.getenv("LLM_MODEL")
    except:
        if mistral:
            llm_model = "mistral-large-latest"
            print("Model set to Mistral-large-latest")
        else:
            llm_model = "deepseek-chat"
            print("Model set to Deepseek-chat")


    if mistral:
        retrieval_strategy_agent: RetrievalStrategyAgent = RetrievalStrategyAgentMistral(llm_base_url, llm_api_key, llm_model)
        generation_agent: GenerationAgent = GenerationAgentMistral(llm_base_url, llm_api_key, llm_model)
        feedback_agent: FeedbackAgent = FeedbackAgentMistral(llm_base_url, llm_api_key, llm_model)
        refinement_agent: RefinementAgent = RefinementAgentMistral(llm_base_url, llm_api_key, llm_model)
        scoring_agent: ScoringAgent = ScoringAgentMistral(llm_base_url, llm_api_key, llm_model)
    else:
        retrieval_strategy_agent: RetrievalStrategyAgent = RetrievalStrategyAgent(llm_base_url, llm_api_key, llm_model)
        generation_agent: GenerationAgent = GenerationAgent(llm_base_url, llm_api_key, llm_model)
        feedback_agent: FeedbackAgent = FeedbackAgent(llm_base_url, llm_api_key, llm_model)
        refinement_agent: RefinementAgent = RefinementAgent(llm_base_url, llm_api_key, llm_model)
        scoring_agent: ScoringAgent = ScoringAgent(llm_base_url, llm_api_key, llm_model)
    
    
    

    # Open the query txt file and read its contents into a string
    with open(query_input_path, 'r') as file:
        query_input = file.read()

    orchestrator_agent_response: OrchestratorAgentResponse = OrchestratorAgentResponse(query_input)
    retrieval_strategy = retrieval_strategy_agent.run(query_input)
    print(retrieval_strategy)
    iteration_cnt = 0
    max_score = None

    while (max_score is None or max_score < score_threshold) and iteration_cnt < iteration_threshold:
        iteration_cnt += 1

        initial_query_response: str = generation_agent.run(work_directory_path, retrieval_strategy, query_input)
        feedback_response: str = feedback_agent.run(work_directory_path, query_input, initial_query_response)
        refined_query_response: str = refinement_agent.run(work_directory_path, query_input, initial_query_response, feedback_response)
        score: float = scoring_agent.run(work_directory_path, query_input, refined_query_response)

        if orchestrator_agent_response.get_total_iteration_cnt() == 0 or score >= orchestrator_agent_response.get_best_response().get_score():
            max_score = score

        new_response = RAGQueryResponse(query_input, refined_query_response, score, iteration_cnt)

        orchestrator_agent_response.add_response(new_response)

    return orchestrator_agent_response

def run_orchestrator_agent_alt(work_directory_path: str, query_input_path: str, query_output_path: str, score_threshold: float = 4, iteration_threshold: int = 3, mistral: bool = False) -> OrchestratorAgentResponse:
    """
    The alternate version of orchestrator agent (without the use of Query Agent). Loop until the score is greater than or equals to the threshold or the iteration count exceeds the limit.

    :param query_input_path: The path to the input query txt to be processed.
    :param score_threshold: The threshold for the score to stop iterating.
    :param iteration_threshold: The maximum number of iterations allowed.

    :return: OrchestratorAgentResponse containing the results.
    """

    try:
        llm_base_url = os.getenv("LLM_BASE_URL")
    except:
        raise Exception("LLM_BASE_URL environment variable is not set.")

    try:
        llm_api_key = os.getenv("LLM_API_KEY")
    except:
        raise Exception("LLM_API_KEY environment variable is not set.")
    
    try:
        llm_model = os.getenv("LLM_MODEL")
    except:
        if mistral:
            llm_model = "mistral-large-latest"
            print("Model set to Mistral-large-latest")
        else:
            llm_model = "deepseek-chat"
            print("Model set to Deepseek-chat")

    if mistral:
        generation_agent: GenerationAgent = GenerationAgentMistral(llm_base_url, llm_api_key)
        feedback_agent: FeedbackAgent = FeedbackAgentMistral(llm_base_url, llm_api_key)
        refinement_agent: RefinementAgent = RefinementAgentMistral(llm_base_url, llm_api_key)
        scoring_agent: ScoringAgent = ScoringAgentMistral(llm_base_url, llm_api_key)
    else:
        generation_agent: GenerationAgent = GenerationAgent(llm_base_url, llm_api_key)
        feedback_agent: FeedbackAgent = FeedbackAgent(llm_base_url, llm_api_key)
        refinement_agent: RefinementAgent = RefinementAgent(llm_base_url, llm_api_key)
        scoring_agent: ScoringAgent = ScoringAgent(llm_base_url, llm_api_key)

    # Open the query txt file and read its contents into a string
    with open(query_input_path, 'r') as file:
        query_input = file.read()

    orchestrator_agent_response: OrchestratorAgentResponse = OrchestratorAgentResponse(query_input)
    iteration_cnt = 0
    max_score = None

    while (max_score is None or max_score < score_threshold) and iteration_cnt < iteration_threshold:
        iteration_cnt += 1

        for retrieval_strategy in ["global", "local", "naive"]:
            initial_query_response: str = generation_agent.run(work_directory_path, retrieval_strategy, query_input)
            feedback_response: str = feedback_agent.run(work_directory_path, query_input, initial_query_response)
            refined_query_response: str = refinement_agent.run(work_directory_path, query_input, initial_query_response, feedback_response)
            score: float = scoring_agent.run(work_directory_path, query_input, refined_query_response)

            if orchestrator_agent_response.get_total_iteration_cnt() == 0 or score >= orchestrator_agent_response.get_best_response().get_score():
                max_score = score
            
            new_response = RAGQueryResponse(query_input, refined_query_response, score, iteration_cnt)

            orchestrator_agent_response.add_response(new_response)

    return orchestrator_agent_response

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

    parser.add_argument(
        '--alternate', 
        action='store_true',
        help='Using alternate multi-agent workflow (without the use of retrieval_strategy_agent)'
    )

    parser.add_argument(
        '--mistral', 
        action='store_true',
        help='Using Mistral AI as LLM'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    work_directory_path = args.work_directory
    query_input_path = args.query_input_path
    query_output_path = args.query_output_path

    if args.alternate:
        run_orchestrator_agent_alt(work_directory_path=work_directory_path, query_input_path=query_input_path, query_output_path=query_output_path, score_threshold=100, iteration_threshold=3, mistral=args.mistral)
    else:
        run_orchestrator_agent(work_directory_path=work_directory_path, query_input_path=query_input_path, query_output_path=query_output_path, score_threshold=100, iteration_threshold=3, mistral=args.mistral)