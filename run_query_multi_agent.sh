#!/bin/bash

python agents/multi_agent_graphrag.py \
    --work_directory /research/d2/msc/khsew24/nano-graphrag/nano_graphrag_cache_llm_TEST_multihop \
    --query_input_path /research/d2/msc/khsew24/nano-graphrag/evaluation/MedQA/input/1.txt \
    --query_output_path /research/d2/msc/khsew24/nano-graphrag/evaluation/MedQA/output/1.md