import os
import argparse

def calculate_accuracy(student_answer_folder, standard_answer_folder):
    # List all files in the folders
    student_files = os.listdir(student_answer_folder)
    standard_answer_files = os.listdir(standard_answer_folder)
    
    # Ensure the files match in both folders
    common_files = set(student_files).intersection(set(standard_answer_files))
    
    if not common_files:
        raise ValueError("No matching files found between the two folders!")
    
    correct_answers = 0
    total_questions = len(common_files)
    
    for file_name in common_files:
        # Read the student's answer
        student_file_path = os.path.join(student_answer_folder, file_name)
        with open(student_file_path, 'r') as sf:
            student_answer = sf.read().strip()
        
        # Read the standard answer
        standard_answer_file_path = os.path.join(standard_answer_folder, file_name)
        with open(standard_answer_file_path, 'r') as stf:
            standard_answer = stf.read().strip()
        
        # Compare answers
        if student_answer == standard_answer:
            correct_answers += 1
    
    # Calculate accuracy
    accuracy = (correct_answers / total_questions) * 100
    return accuracy

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate the accuracy of student answers.")
    parser.add_argument('--student_answer_folder', type=str, required=True, help="Path to the student answer folder")
    parser.add_argument('--standard_answer_folder', type=str, required=True, help="Path to the standard answer folder")

    # Parse the command line arguments
    args = parser.parse_args()

    # Calculate and print accuracy
    accuracy = calculate_accuracy(args.student_answer_folder, args.standard_answer_folder)
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()