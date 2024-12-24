import argparse
import os
from time import time
import glob
from mistralai import Mistral
from openai import AsyncOpenAI
import json
import asyncio
"""
This script is used to compare the responses generated by 1. Without GraphRAG 2. With GraphRAG 3. With Multi-agent GraphRAG
"""

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
    help='The input path of the folder containing the query txt files.'
)

parser.add_argument(
    '--query_output_path',
    type=str,
    required=True,
    help='The output path of the folder containing the responses.'
)

parser.add_argument(
    '--mistral',
    action='store_true',
    help='Using Mistral AI as LLM'
)

# Parse the arguments
args = parser.parse_args()
file_list = glob.glob(args.query_input_path+'/'+'*.txt')
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

async def query_without_rag(QUERY_QUESTION, mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = AsyncOpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    messages.append({"role": "user", "content": QUERY_QUESTION})
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = await client.chat.completions.create(
            model=llm_model, messages=messages
        )
    response: str = chat_response.choices[0].message.content
    return response

async def comparison_agent(question, answer1, answer2, answer3 , mistral):
    if mistral:
        client = Mistral(api_key=llm_api_key)
    else:
        client = AsyncOpenAI(
            api_key=llm_api_key, base_url=llm_base_url
        )
    messages = []
    messages.append({"role": "system", "content": """
                     You are an expert tasked with evaluating two answers to the same question based on three criteria: *Comprehensiveness, **Diversity, and **Empowerment**.
                     ---Goal---
                    You will evaluate two answers to the same question based on three criteria: *Comprehensiveness, **Diversity, and **Empowerment*.

                    - Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
                    - Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
                    - Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

                    For each criterion, choose the better answer (either Answer 1, Answer 2 or Answer 3) and explain why. Then, select an overall winner based on these three categories.
                     """})
    messages.append({"role": "system", "content": f"Here is the question: {question}"})
    messages.append({"role": "user", "content": f"Here is answer 1: {answer1}"})
    messages.append({"role": "user", "content": f"Here is answer 2: {answer2}"})
    messages.append({"role": "user", "content": f"Here is answer 3: {answer3}"})
    messages.append({"role": "user", "content": "Evaluate the above answers using the three criteria listed above and provide detailed explanations for each criterion."})
    messages.append({"role": "user", "content": 
                     """Output your evaluation in the following JSON format:
                        {
                            "Comprehensiveness": {
                                "Winner": "[Answer 1/ Answer 2/ Answer 3]",
                                "Explanation": "[Provide explanation here]"
                            },
                            "Diversity": {
                                "Winner": "[Answer 1/ Answer 2/ Answer 3]",
                                "Explanation": "[Provide explanation here]"
                            },
                            "Empowerment": {
                                "Winner": "[Answer 1/ Answer 2/ Answer 3]",
                                "Explanation": "[Provide explanation here]"
                            },
                            "Overall Winner": {
                                "Winner": "[Answer 1/ Answer 2/ Answer 3]",
                                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
                            }
                        }
                    """
                    })
    if mistral:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=messages
        )
    else:
        chat_response = await client.chat.completions.create(
            model=llm_model, messages=messages
        )
    response = chat_response.choices[0].message.content
    return response

try:
    os.mkdir(f"{args.query_output_path}")
except:
    pass
try:
    os.mkdir(f"{args.query_output_path}/without_rag")
except:
    pass
try:
    os.mkdir(f"{args.query_output_path}/with_rag")
except:
    pass    
try:
    os.mkdir(f"{args.query_output_path}/with_multiagent_rag")
except:
    pass
try:
    os.mkdir(f"{args.query_output_path}/final_comparison")
except:
    pass

timing = {"without_rag": {}, "with_rag": {}, "with_multiagent_rag": {}}

# 1. Run query without the use of GraphRAG
for filename in file_list:
    print(filename)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        start = time()
        file_contents = file.read()
        response = asyncio.run(query_without_rag(f"{file_contents}", args.mistral))
        with open(f"{args.query_output_path}/without_rag/response_{os.path.basename(filename)}", "w") as f:
            print(response, file=f)
        time_taken = time() - start
        timing["without_rag"][filename] = time_taken
        print(f"(Without the use of GraphRAG) query time for {filename}:", time_taken)

# 2. Run query with the use of GraphRAG
for filename in file_list:
    print(filename)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        try:
            os.mkdir(f"{args.query_output_path}/with_rag/response_{filename}")
        except:
            pass 
        # Read the contents of the file
        start = time()
        if args.mistral:
            os.system(f"python ./examples/using_mistral_as_llm+ollama_embedding.py --run_query --working_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_rag/response_{os.path.basename(filename)}")
        else:
            os.system(f"python ./examples/using_llm_api_as_llm+ollama_embedding.py --run_query --working_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_rag/response_{os.path.basename(filename)}")
        time_taken = time() - start
        timing["with_rag"][filename] = time_taken
        print(f"(With the use of GraphRAG) query time for {filename}:", time_taken)

# 3. Run query with the use of Multi-agent GraphRAG
for filename in file_list:
    print(filename)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        try:
            os.mkdir(f"{args.query_output_path}/with_rag/response_{filename}")
        except:
            pass 
        if args.mistral:
            try:
                start = time()
                os.system(f"python ./examples/multi_agent_graphrag_parser.py --work_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_multiagent_rag/response_{os.path.basename(filename)} --mistral")
            except:
                start = time()
                os.system(f"python ./examples/multi_agent_graphrag_parser.py --work_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_multiagent_rag/response_{os.path.basename(filename)} --mistral")
        else:
            try:
                start = time()
                os.system(f"python ./examples/multi_agent_graphrag_parser_async.py --work_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_multiagent_rag/response_{os.path.basename(filename)}")
            except:
                start = time()
                os.system(f"python ./examples/multi_agent_graphrag_parser_async.py --work_directory {args.work_directory} --query_input_path {filename} --query_output_path {args.query_output_path}/with_multiagent_rag/response_{os.path.basename(filename)}")
        time_taken = time() - start
        timing["with_multiagent_rag"][filename] = time_taken
        print(f"(With the use of Multi-agent GraphRAG) query time for {filename}:", time_taken)

# 4. Comparison of the results
file_list = glob.glob(args.query_input_path+'/'+'*.txt')
for filename in file_list:
    print(filename)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        question = file.read()
    with open(f"{args.query_output_path}/without_rag/response_{os.path.basename(filename)}", 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        answer1 = file.read()
    with open(f"{args.query_output_path}/with_rag/response_{os.path.basename(filename)}", 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        answer2 = file.read()
    with open(f"{args.query_output_path}/with_multiagent_rag/response_{os.path.basename(filename)}", 'r', encoding='utf-8-sig') as file:
        # Read the contents of the file
        answer3 = file.read()
    result = asyncio.run(comparison_agent(question, answer1, answer2, answer3 , args.mistral))
    try:
        with open(f"{args.query_output_path}/final_comparison/response_{os.path.basename(filename)}.json", 'w', encoding='utf-8-sig') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    except:
        with open(f"{args.query_output_path}/final_comparison/response_{os.path.basename(filename)}.txt", 'w', encoding='utf-8-sig') as f:
            f.write(result)
    with open(f"{args.query_output_path}/final_comparison/response_time.txt", 'w') as f:
        json.dump(timing, f)