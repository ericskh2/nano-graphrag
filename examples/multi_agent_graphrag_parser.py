import os
import logging
from typing import Literal
import argparse
import ollama
import numpy as np
from mistralai import Mistral
from openai import OpenAI, AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import glob
import re
import time

# logger settings
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Assumed embedding model settings
embedding_model_name = "nomic-embed-text:ctx32k"
embedding_model_dim = 768
embedding_model_max_tokens = 32000

# Mistral LLM Model
SYSTEM_PROMPT_TEMPLATE = "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example. You must enclose your respond in JSON object."
async def llm_model_mistral_if_cache(
        prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE, history_messages=[], **kwargs
) -> str:
    client = Mistral(api_key=llm_api_key)
    messages = []
    if system_prompt:
    # system_prompt = "You are a answer agent to answer a question."
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # if hashing_kv is not None:
    #     args_hash = compute_args_hash(model, messages)
    #     if_cache_return = await hashing_kv.get_by_id(args_hash)
    #     if if_cache_return is not None:
    #         return if_cache_return["return"]
    # -----------------------------------------------------

    # print("kwargs",kwargs)
    response = client.chat.complete(
        model=llm_model, messages=messages
    )

    # Cache the response if having-------------------
    # if hashing_kv is not None:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": model}}
    #     )
    # -----------------------------------------------------
    return response.choices[0].message.content


SYSTEM_PROMPT_TEMPLATE = "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example. You must enclose your respond in JSON object."
async def llm_model_deepseek_if_cache(
        prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=llm_api_key, base_url=llm_base_url
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    # hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # if hashing_kv is not None:
    #     args_hash = compute_args_hash(MODEL, messages)
    #     if_cache_return = await hashing_kv.get_by_id(args_hash)
    #     if if_cache_return is not None:
    #         return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=llm_model, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    # if hashing_kv is not None:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
    #     )
    # -----------------------------------------------------

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


# Use Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=embedding_model_dim,
    max_token_size=embedding_model_max_tokens,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=embedding_model_name, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text

# Insert
# def insert(File_Directory):
#     from time import time

#     remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
#     remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
#     remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
#     remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
#     remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

#     rag = GraphRAG(
#         working_dir=WORKING_DIR,
#         enable_llm_cache=True,
#         best_model_func=llm_model_mistral_if_cache,
#         cheap_model_func=llm_model_mistral_if_cache,
#         embedding_func=ollama_embedding,
#     )
#     start = time()
#     file_list = glob.glob(File_Directory + '/' + 'book.txt')
#     for filename in file_list:
#         with open(filename, 'r', encoding='utf-8-sig') as file:
#             # Read the contents of the file
#             file_contents = file.read()
#             rag.insert(file_contents)

#     print("indexing time:", time() - start)


# Agent Works Here
def query_retrieval_strategy(QUERY_QUESTION, mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = OpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    system_prompt = "You are an intelligent agent responsible for determining the appropriate query type based on the provided scenario. Your query types include local, global, and naive queries. I will give you a question and you need to analyze the question and choose the best query type to obtain the required information efficiently. Please give the answer only."
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": QUERY_QUESTION})
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = client.chat.completions.create(
            model=llm_model, messages=messages
        )
    print('Running retrieval strategy agent')
    result: Literal["local","global","naive"] = "global"
    response: str = chat_response.choices[0].message.content
    if response.lower().find("local") > 0 :
        result = "local"
    elif response.lower().find("global") > 0 :
        result = "global"
    elif response.lower().find("naive") > 0 :
        result = "naive"
    return result

def scoring_alt(QUERY_QUESTION, mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = OpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    system_prompt = "You are an intelligent agent responsible for scoring the response by another agent."
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": QUERY_QUESTION})
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = client.chat.completions.create(
            model=llm_model, messages=messages
        )
    print('Running alternate scoring agent')
    response: str = chat_response.choices[0].message.content
    return response

def feedback_alt(QUERY_QUESTION, mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = OpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    system_prompt = "You are an intelligent agent responsible for giving feedback of responses from another agent."
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": QUERY_QUESTION})
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = client.chat.completions.create(
            model=llm_model, messages=messages
        )
    print('Running alternate feedback agent')
    response: str = chat_response.choices[0].message.content
    return response

def refinement_alt(QUERY_QUESTION, mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = AsyncOpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    system_prompt = "You are an intelligent agent responsible for refining the response based on the feedback provided by another agent."
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": QUERY_QUESTION})
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = client.chat.completions.create(
            model=llm_model, messages=messages
        )
    print('Running alternate refinement agent')
    response: str = chat_response.choices[0].message.content
    return response

def query_graphrag(QUERY_QUESTION, mode, mistral):
    if mistral:
        rag = GraphRAG(
            working_dir=WORKING_DIR,
            best_model_func=llm_model_mistral_if_cache,
            cheap_model_func=llm_model_mistral_if_cache,
            embedding_func=ollama_embedding,
        )
        response = rag.query(
            QUERY_QUESTION,
            param=QueryParam(mode=mode)
        )
    else:
        rag = GraphRAG(
            working_dir=WORKING_DIR,
            best_model_func=llm_model_deepseek_if_cache,
            cheap_model_func=llm_model_deepseek_if_cache,
            embedding_func=ollama_embedding,
        )
        response = rag.query(
            QUERY_QUESTION,
            param=QueryParam(mode=mode)
        )
    return response

if __name__ == '__main__':
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

    # parser.add_argument(
    #     '--alternate',
    #     action='store_true',
    #     help='Using alternate multi-agent workflow (without the use of retrieval_strategy_agent)'
    # )

    parser.add_argument(
        '--mistral',
        action='store_true',
        help='Using Mistral AI as LLM'
    )

    parser.add_argument(
        '--alternate_agent',
        action='store_true',
        help='Using non Graph-RAG Feedback, Refinement and Scoring Agent'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    WORKING_DIR = args.work_directory
    INPUT_PATH = args.query_input_path
    OUTPUT_PATH = args.query_output_path

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
        if args.mistral:
            llm_model = "mistral-large-latest"
            print("Model set to Mistral-large-latest")
        else:
            llm_model = "deepseek-chat"
            print("Model set to Deepseek-chat")

    with open(INPUT_PATH, 'r') as file:
        QUERY_QUESTION = file.read()

    try:
        retrieval_strategy = query_retrieval_strategy(QUERY_QUESTION, args.mistral)
        print("retrieval_strategy", retrieval_strategy)
    except:
        time.sleep(1)
        retrieval_strategy = query_retrieval_strategy(QUERY_QUESTION, args.mistral)
        print("retrieval_strategy", retrieval_strategy)

    try:
        generated_response = query_graphrag(QUERY_QUESTION, retrieval_strategy, args.mistral)
        print("generated_response", generated_response)
    except:
        time.sleep(1)
        generated_response = query_graphrag(QUERY_QUESTION, retrieval_strategy, args.mistral)
        print("generated_response", generated_response)

    score_threshold: float = 4.0
    iteration_threshold: int = 3
    max_score: float = 0.0
    iteration_cnt: int = 0
    best_response = None
    while iteration_cnt < iteration_threshold:
        iteration_cnt += 1

        FeedBack_Prompt = f"""
                Here is the question: "{QUERY_QUESTION}".
                Here is the answer: "{generated_response}".
                If the question is open-ended, please give your feedback on the answer and give the advantages and disadvantages of the answer, longer answers are accepted.
                If the question is multiple-choice type, please verify the answer and give the reasons."
                If the question is a short question, please give your feedback on the answer and verify its correctness, longer answers are not accepted.
            """
        try:
            if args.alternate_agent:
                feedback = feedback_alt(FeedBack_Prompt, args.mistral)
            else:
                feedback = query_graphrag(FeedBack_Prompt, retrieval_strategy, args.mistral)
            print("feedback", feedback)
        except:
            time.sleep(1)
            if args.alternate_agent:
                feedback = feedback_alt(FeedBack_Prompt, args.mistral)
            else:
                feedback = query_graphrag(FeedBack_Prompt, retrieval_strategy, args.mistral)
            print("feedback", feedback)

        Refinement_Prompt = f"""
               Here is the question: "{QUERY_QUESTION}".
               This is the initial response: "{generated_response}".
               This is the feedback/verfication of the initial response: "{feedback}".
               If the question is open-ended, please refine the response of the question based on the feedback by highlighting the advantages and addressing the disadvantages mentioned.
               If the question is multiple-choice type, please revise the answer based on the feedback and return the correct answer (a single letter only, such as: A, B, C, D or E).
               If the question is a short question, only return the answer directly based on the feedback.
            """
        try:
            if args.alternate_agent:
                refined_response = refinement_alt(Refinement_Prompt, args.mistral)
            else:
                refined_response = query_graphrag(Refinement_Prompt, retrieval_strategy, args.mistral)
            print("refined_response", refined_response)
        except:
            time.sleep(1)
            if args.alternate_agent:
                refined_response = refinement_alt(Refinement_Prompt, args.mistral)
            else:
                refined_response = query_graphrag(Refinement_Prompt, retrieval_strategy, args.mistral)
            print("refined_response", refined_response)

        Scoring_Prompt = f"""
                Here is the question: "{QUERY_QUESTION}".
                Here is the refined answer: "{refined_response}".
                Give a lower score if the answer does not meet the following requirement: a multiple-choice question should not have any explanations, only the answer letter; a short question should have a concise answer; an open-ended question should have a detailed answer.
                Please return a score only from 1-5 (type: float) about whether the refined response is satisfactory. You must score fairly and I am able to tolerate low scores. Only a single floating point number is allowed in the response.
            """
        try:
            if args.alternate_agent:
                score = scoring_alt(Scoring_Prompt, args.mistral)
                match = re.search(r"[-+]?\d*\.\d+|\d+", score)
                if match:
                    score = float(match.group())
            else:
                score = query_graphrag(Scoring_Prompt, retrieval_strategy, args.mistral)
                match = re.search(r"[-+]?\d*\.\d+|\d+", score)
                if match:
                    score = float(match.group())
        except:
            time.sleep(1)
            if args.alternate_agent:
                score = scoring_alt(Scoring_Prompt, args.mistral)
                match = re.search(r"[-+]?\d*\.\d+|\d+", score)
                if match:
                    score = float(match.group())
            else:
                score = query_graphrag(Scoring_Prompt, retrieval_strategy, args.mistral)
                match = re.search(r"[-+]?\d*\.\d+|\d+", score)
                if match:
                    score = float(match.group())
        print("score", score)

        if score > max_score:
            max_score = score
            best_response = refined_response
            generated_response = best_response

    print("best_response", best_response)
    print("best_score", max_score)

    with open(OUTPUT_PATH, 'w') as file:
        file.write(
            f"""
            Response:
            {best_response}
            Score: {max_score}
            """
        )


