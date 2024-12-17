#!/bin/bash

python examples/using_llm_api_as_llm+ollama_embedding.py \
    --run_insert \
    --documents_path /research/d2/msc/khsew24/nano-graphrag/evaluation/MultiHop-RAG/dataset/corpus \
    --input_path /research/d2/msc/khsew24/nano-graphrag/evaluation/MedQA/input/1.txt \
    --output_path /research/d2/msc/khsew24/nano-graphrag/evaluation/MedQA/output/1.md