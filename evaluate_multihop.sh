#!/bin/bash

# Start the timer
SECONDS=0

# Set mode: rag/non-rag
mode="rag"

# Prerequisite: Prepare work directory (inserted knowledge graph), raw queries and questiontype txts

working_dir="Dec19_nano_graphrag_cache_llm_TEST_multihop"
query_input_dir="evaluation/MultiHop-RAG/dataset/rawqueries"
questiontype_dir="evaluation/MultiHop-RAG/dataset/questiontypes"

response_output_dir="evaluation/MultiHop-RAG/output/rag_output"
processed_response_output_json_path="evaluation/MultiHop-RAG/output/processed_rag_output.json"
standard_answer_json_path="evaluation/MultiHop-RAG/dataset/MultiHopRAG.json"

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
        --query_output_path "$output_file" \
        --prompt_mode noexplanation
  else
    echo "Running non-rag mode"
    python examples/query_non_rag.py \
    --query_input_path "$input_file" \
    --query_output_path "$output_file" \
    --prompt_mode noexplanation
  fi
done

Process response from RAG
python evaluation/MultiHop-RAG/response_process.py \
  --queries_folder_path $query_input_dir \
  --response_folder_path $response_output_dir \
  --questiontypes_folder_path $questiontype_dir \
  --output_json_path $processed_response_output_json_path

# Calculate accuracy using the second Python script
python evaluation/MultiHop-RAG/qa_evaluate.py \
  --response_output_json_path $processed_response_output_json_path \
  --standard_answer_json_path $standard_answer_json_path

# Calculate elapsed time
elapsed_time=$SECONDS
echo "Elapsed time: $elapsed_time seconds"
