#!/bin/bash

python examples/looporder_response_comparison_async.py \
    --work_directory nano_graphrag_cache_llm_TEST_web3 \
    --query_input_path evaluation/web3-smaller/web3_raw_questions \
    --query_output_path evaluation/web3/responses
