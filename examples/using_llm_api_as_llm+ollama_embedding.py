import os
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import glob
import os

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Setting
RETRIEVE_AGAIN = False
# QUERY_QUESTION = "What is web3?"
# QUERY_QUESTION = "Provide a summary of the development history of web3"
QUERY_QUESTION = "How does Bitcoin enhance the privacy level of transactions when compared to the traditional banking model?​"

# Path setup
web3_text_path = "/research/d2/msc/khsew24/cryptoKGTutorial/txtWhitePapers/*.txt"

# Assumed llm model settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL = "deepseek-chat"

# Assumed embedding model settings
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192


async def llm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # if hashing_kv is not None:
    #     args_hash = compute_args_hash(MODEL, messages)
    #     if_cache_return = await hashing_kv.get_by_id(args_hash)
    #     if if_cache_return is not None:
    #         return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    # if hashing_kv is not None:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
    #     )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


WORKING_DIR = "./nano_graphrag_cache_llm_TEST"


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )
    print(
        rag.query(
            QUERY_QUESTION, param=QueryParam(mode="global")
        )
    )


def insert():
    from time import time

    # with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
    #     FAKE_TEXT = f.read()

    file_contents_list = []
    file_list = glob.glob(web3_text_path)
    for filename in file_list:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            # Read the contents of the file
            file_contents = file.read()
            file_contents_list.append(file_contents)

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    rag.insert(file_contents_list)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])

# We're using Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim= EMBEDDING_MODEL_DIM,
    max_token_size= EMBEDDING_MODEL_MAX_TOKENS,
)

async def ollama_embedding(texts :list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
      data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
      embed_text.append(data["embedding"])
    
    return embed_text

if __name__ == "__main__":
    if RETRIEVE_AGAIN:
        insert()
    query()
