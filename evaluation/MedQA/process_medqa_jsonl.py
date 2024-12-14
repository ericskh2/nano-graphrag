import json
import os
import shutil

# Path to the JSONL file
jsonl_file_path = "data_clean/questions/US/test.jsonl"

# txt output folder
output_folder = "input" # script's output = input folder for query

# Remove the folder if it exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Create the folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

"""
question - A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?
answer - Tell the attending that he cannot fail to disclose this mistake
options - {'A': 'Disclose the error to the patient but leave it out of the operative report', 'B': 'Disclose the error to the patient and put it in the operative report', 'C': 'Tell the attending that he cannot fail to disclose this mistake', 'D': 'Report the physician to the ethics committee', 'E': 'Refuse to dictate the operative report'}
meta_info - step1
answer_idx - C
"""

# Reading the file
questions = []
with open(jsonl_file_path, "r") as file:
    for line in file:
        qa = json.loads(line)
        questions.append({
            'question': qa['question'],
            'options': qa['options'],
            'answer_idx': qa['answer_idx']
        })

question_idx = 1
for q in questions:
    prompt = f"You are a multiple-choice question answering assistant. Here is a question and its choices: Question: {q['question']} Options: {q['options']} Respond with the correct choice's letter (A/B/C/D) only, without any explanation."
    # Open a file in write mode ('w'), or create the file if it doesn't exist
    with open(os.path.join(output_folder, f'{question_idx}.txt'), 'w') as file:
        file.write(prompt)
    question_idx += 1