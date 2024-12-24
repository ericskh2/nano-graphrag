import os
import logging
from typing import Literal

import ollama
import numpy as np
from mistralai import Mistral
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import glob

# logger settings
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Assumed llm model settings - Mistral
llm_api_key_mistral = "WkR7n3wrHUFz02P7NgjweucocW2yIRFZ"
llm_model_name_mistral = "mistral-large-latest"

# Assumed llm model settings - Deepseek
llm_base_url_deepseek = os.getenv("LLM_BASE_URL")
llm_api_key_deepseek = os.getenv("LLM_API_KEY")
llm_model_name_deepseek = os.getenv("LLM_MODEL")

# Assumed embedding model settings
embedding_model_name = "nomic-embed-text"
embedding_model_dim = 768
embedding_model_max_tokens = 8192

# Mistral LLM Model
async def llm_model_mistral_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = Mistral(api_key=llm_api_key_mistral)
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
        model=llm_model_name_mistral, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    # if hashing_kv is not None:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": model}}
    #     )
    # -----------------------------------------------------
    return response.choices[0].message.content


SYSTEM_PROMPT_TEMPLATE = "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."
async def llm_model_deepseek_if_cache(
        prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=llm_api_key_deepseek, base_url=llm_base_url_deepseek
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
        model=llm_model_name_deepseek, messages=messages, **kwargs
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


WORKING_DIR = "./nano_graphrag_cache_mistral_test2"
# Insert
def insert(File_Directory):
    from time import time

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=llm_model_mistral_if_cache,
        cheap_model_func=llm_model_mistral_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    file_list = glob.glob(File_Directory + '/' + 'book.txt')
    for filename in file_list:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            # Read the contents of the file
            file_contents = file.read()
            rag.insert(file_contents)

    print("indexing time:", time() - start)


# Agent Works Here
def query_retrieval_strategy(QUERY_QUESTION):
    client = Mistral(api_key=llm_api_key_mistral)
    messages = []
    system_prompt = "You are an intelligent agent responsible for determining the appropriate query type based on the provided scenario. Your query types include local, global, and naive queries. I will give you a question and you need to analyze the question and choose the best query type to obtain the required information efficiently. Please give the answer only."
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": QUERY_QUESTION})
    chat_response = client.chat.complete(
        model=llm_model_name_mistral,
        messages=messages
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


def query_graphrag(QUERY_QUESTION, mode):
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
    return response


if __name__ == "__main__":
    inserted : bool = False
    if inserted == True:
        File_Directory = "D:/Pycharm/graphrag-multiagent/docs"
        insert(File_Directory)

    QUERY_QUESTION = "What are the top themes in the story?"
    retrieval_strategy = query_retrieval_strategy(QUERY_QUESTION)
    print("retrieval_strategy", retrieval_strategy)

    generated_response = query_graphrag(QUERY_QUESTION, retrieval_strategy)
    print("generated_response", generated_response)

    score_threshold: float = 4.0
    iteration_threshold: int = 3
    max_score: float = 0.0
    iteration_cnt: int = 0
    best_response = None
    while max_score < score_threshold and iteration_cnt < iteration_threshold:
        iteration_cnt += 1

        FeedBack_Prompt = f"""
            Here is the question: "{QUERY_QUESTION}".
            Here is the answer: "{generated_response}".
            Please give your feedback on the answer and give the advantages and disadvantages of the answer"
        """
        feedback = query_graphrag(FeedBack_Prompt, retrieval_strategy)
        print("feedback", feedback)

        Refinement_Prompt = f"""
           Here is the question: "{QUERY_QUESTION}".
           This is the initial response: "{generated_response}".
           This is the feedback of the initial response: "{feedback}".
           Please refine the response of the question based on the feedback by highlighting the advantages and addressing the disadvantages mentioned.
        """
        refined_response = query_graphrag(Refinement_Prompt, retrieval_strategy)
        print("refined_response", refined_response)

        Scoring_Prompt = f"""
            Here is the question: "{QUERY_QUESTION}".
            Here is the refined answer: "{refined_response}".
            Please give a score only from 1-5(type: float) about whether the refined response is satisfactory. Please just give the score only and you must score fairly without worrying about appearances and I am able to tolerate low scores.
        """
        score = query_graphrag(Scoring_Prompt, retrieval_strategy)
        print("score", score)

        if score > max_score:
            max_score = score
            best_response = refined_response
            generated_response = best_response

    print("best_response", best_response)
    print("best_score", max_score)



