#!/bin/bash

# Set mode: rag/non-rag
mode="non-rag" # Change to "Non-RAG" for non-RAG mode

working_dir="nano_graphrag_cache_llm_TEST_medqa/"
query_input_dir="evaluation/MedQA/input/"
response_output_dir="evaluation/MedQA/output/"
standard_answer_dir="evaluation/MedQA/answers/"

# Ensure output directory exists
mkdir -p "$response_output_dir"

# Loop through all .txt files in the input directory
for input_file in "$query_input_dir"/*.txt; do
  # Extract the filename without extension
  filename=$(basename "$input_file" .txt)
  
  # Define the corresponding output file path
  output_file="$response_output_dir/$filename.txt"

  if [ "$mode" == "rag" ]; then
    # Run the Python script with the current input and output paths
    python examples/using_llm_api_as_llm+ollama_embedding.py \
        --run_query \
        --working_directory "$working_dir" \
        --query_input_path "$input_file" \
        --query_output_path "$output_file"
  else
    python examples/query_non_rag.py \
    --query_input_path "$input_file" \
    --query_output_path "$output_file"
  fi
done

# Calculate accuracy using the second Python script
python evaluation/MedQA/calculate_medqa_accuracy.py \
  --student_answer_folder "$response_output_dir" \
  --standard_answer_folder "$standard_answer_dir"