import os
import argparse
from openai import AsyncOpenAI
import asyncio  # Required to run the async function

# Assumed llm model settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL = "deepseek-chat"

async def call_llm_api(prompt, **kwargs) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )

    messages = []
    messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content

async def query(query_input_path: str, query_output_path: str, prompt_mode=None):
    # Open the file in read mode and read its contents into a string
    with open(query_input_path, "r") as file:
        query_question = file.read()  # Read the entire file as a single string
    
    if prompt_mode == 'explanation':
        prompt_content = f'Answer the following question and provide a detailed explanation for your answer:: {query_question}'
    elif prompt_mode == 'noexplanation':
        prompt_content = f'You are a question answering assistant. You will be given a question. The answer to the question is a word or entity. Answer directly without punctuation and without explanation: {query_question}'
    elif prompt_mode == 'multiplechoice':
        prompt_content = f"You are a multiple-choice question answering assistant. You will be given a question and its choices. Respond with the correct choice's letter (A/B/C/D/E/F) only, without any explanation: {query_question}"
    else:
        raise Exception(f'Unknown prompt_mode {prompt_mode}!')
    
    print(prompt_content)

    response = await call_llm_api(prompt=prompt_content)

    # Open the file in write mode
    with open(query_output_path, "w") as file:
        # Use the print function with the file parameter
        print(
            response,
            file=file
        )

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Script to process files with input and output paths.")

    # Add input and output arguments
    parser.add_argument('--query_input_path', type=str, help='Path to the input txt file that contains the question')
    parser.add_argument('--query_output_path', type=str, help='Path to the output file tht outputs the answer')
    parser.add_argument('--prompt_mode', type=str, help='Prompt mode (explanation, noexplanation, multiplechoice)')

    # Parse the arguments
    args = parser.parse_args()

    asyncio.run(query(query_input_path=args.query_input_path, query_output_path=args.query_output_path, prompt_mode=args.prompt_mode))