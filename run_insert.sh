#!/bin/bash

# Infinite loop
while true; do
    python examples/using_llm_api_as_llm+ollama_embedding.py \
        --run_insert \
        --working_directory /research/d2/msc/khsew24/nano-graphrag-medqa/nano_graphrag_cache_llm_TEST_medqa/ \
        --documents_path /research/d2/msc/khsew24/nano-graphrag-medqa/evaluation/MedQA/data_clean/textbooks/en/ \
    # Add a delay to prevent rapid looping
    sleep 1
done
