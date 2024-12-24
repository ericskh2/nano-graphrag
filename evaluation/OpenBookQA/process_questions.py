import json
import os

# Specify the input JSONL file
input_file = 'data/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl'  # Replace with your JSONL file name
queries_output_dir = 'data/rawqueries'
ans_output_dir = 'data/answers'

os.makedirs(queries_output_dir, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(ans_output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Read and process the JSONL file
with open(input_file, 'r') as file:

    idx = 0
    for line in file:
        # Parse the JSON object from the line
        json_object = json.loads(line.strip())

        q = json_object['question']
        ans = json_object['answerKey']

        # print('q',q)
        # print('ans',ans)

        choices_content = ' '.join(f"{c['label']}: {c['text']}" for c in q['choices'])

        # Write query to a text file
        query_content = f"Question: {q['stem']}; Choices: {choices_content}"
        query_output_path = os.path.join(queries_output_dir, f'{idx + 1}.txt')
        with open(query_output_path, 'w') as qoutfile:
            qoutfile.write(query_content)

        # Write answer to a text file
        answer_output_path = os.path.join(ans_output_dir, f'{idx + 1}.txt')
        with open(answer_output_path, 'w') as outfile:
            outfile.write(ans)  # Write as pretty JSON

        idx += 1