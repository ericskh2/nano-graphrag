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
import argparse

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# QUERY_QUESTION = "What is web3?"
# QUERY_QUESTION = "Provide a summary of the development history of web3"
# QUERY_QUESTION = "How does Bitcoin enhance the privacy level of transactions when compared to the traditional banking model?​"
QUERY_QUESTION = None

# Path setup
# web3_text_path = "/research/d2/msc/khsew24/cryptoKGTutorial/txtWhitePapers/*.txt"

# Assumed llm model settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL = "deepseek-chat"

# Assumed embedding model settings
EMBEDDING_MODEL = "nomic-embed-text:ctx32k"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 32000

SYSTEM_PROMPT_TEMPLATE = "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."

WORKING_DIR = None
# WORKING_DIR = "./nano_graphrag_cache_llm_TEST_multihop"

async def llm_model_if_cache(
    prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
    messages = []
    if system_prompt:
        # print('YES')
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    # print(messages)
    
    # print('-'*10)

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)




def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )

    # Open the file in write mode
    with open(args.query_output_path, "w") as file:
        # Use the print function with the file parameter
        print(
            rag.query(
                QUERY_QUESTION, param=QueryParam(mode="global")
            ),
            file=file
        )
        
def insert(documents_directory_path):
    from time import time

    # with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
    #     FAKE_TEXT = f.read()

    # remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    # remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    # rag.insert(file_contents_list)

    # file_contents_list = []
    file_list = glob.glob(documents_directory_path+'/'+'*.txt')
    for filename in file_list:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            print(f'using_llm_api_as_llm+ollama_embedding.py: Inserting document {filename}')
            # Read the contents of the file
            file_contents = file.read()

            # rag.insert(file_contents)
            # file_contents_list.append(file_contents)

            # Insert the content in chunks
            def insert_in_chunks(content, insert_function, max_chunk_size):
                """
                Inserts content into the graph function in chunks of at most max_chunk_size.

                Args:
                    content (str): The content to be inserted.
                    insert_function (callable): The function to call for inserting chunks.
                    max_chunk_size: Maximum limit of the size of a chunk
                """
                start = 0
                while start < len(content):
                    end = min(start + max_chunk_size, len(content))
                    print(f'using_llm_api_as_llm+ollama_embedding.py: Inserting document {filename} (start:{start}, end:{end})')
                    insert_function(content[start:end])
                    start = end

            insert_in_chunks(file_contents, rag.insert, 2000 * 1024)

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
    # Create the parser
    parser = argparse.ArgumentParser(description="Script to process files with input and output paths.")

    # Add input and output arguments
    parser.add_argument('--run_insert', action='store_true', help='Whether to run insert() again')
    parser.add_argument('--run_query', action='store_true', help='Whether to run query()')
    parser.add_argument('--working_directory', type=str, required=True, help='Path to the work directory')
    parser.add_argument('--documents_path', type=str, help='Path to the document directory that contains corpus txt files')
    parser.add_argument('--query_input_path', type=str, help='Path to the input txt file that contains the question')
    parser.add_argument('--query_output_path', type=str, help='Path to the output file tht outputs the answer')
    
    # Parse the arguments
    args = parser.parse_args()
    
    WORKING_DIR = args.working_directory
    
    print(f'args.run_insert={args.run_insert}')
    if args.run_insert:
        insert(args.documents_path)

    print(f'args.run_query={args.run_query}')
    if args.run_query:
        # Open the file in read mode and read its contents into a string
        with open(args.query_input_path, "r") as file:
            QUERY_QUESTION = file.read()  # Read the entire file as a single string
        
        print(f'QUERY_QUESTION read from file: {QUERY_QUESTION}')
        query()
