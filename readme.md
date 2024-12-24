# CSCI5120 Project - Multi-agent Graph RAG and Evaluation

Group members: CHUNG Wai Kei, CUI Wenqian, PAN Chengjie,  SEW Kin Hang

Acknowledgement: Part of the code is modified from https://github.com/gusye1234/nano-graphrag and https://github.com/yixuantt/MultiHop-RAG

## Installation

Setup conda environment
```
conda env create --name custom_env_name -f environment.yml
```

Download Ollama from https://ollama.com/ or https://github.com/ollama/ollama/releases

Pull new embedding model 
```
ollama pull nomic-embed-text 
```

If intend to run with Mistral AI (Optional)
```
pip install mistralai
```


## Usage
### Setup ollama

Start ollama model (Linux with Command-line Interface)
```
./ollama-linux-amd64/bin/ollama serve& 
export no_proxy=localhost,127.0.0.0,127.0.0.1,127.0.1.1 
```
or run the ollama application (for Windows and MacOS)

### Create Embedding Model (Required for first time usage)
```
ollama show --modelfile nomic-embed-text > Modelfile
```

Add a new line into this `Modelfile` below the 'FROM':

`PARAMETER num_ctx 32000`

```
ollama create -f Modelfile nomic-embed-text:ctx32k
```

### Setup environment variables for LLM API usage
```
cd <Path to nano-graphrag>
export LLM_BASE_URL="https://api.deepseek.com" // or other api
export LLM_API_KEY="YOUR-KEY" 
export LLM_MODEL="model-name" 
```

### Insert documents

(For non-Mistral API usage)
```
python examples/using_llm_api_as_llm+ollama_embedding.py \
    --run_insert \
    --working_directory <Path to store the GraphRAG Backend> \
    --documents_path <Path to document folder containing txt files> 
```
(For Mistral API usage)
```
python examples/using_mistral_api_as_llm+ollama_embedding.py \
    --run_insert \
    --working_directory <Path to store the GraphRAG Backend> \
    --documents_path <Path to document folder containing txt files> 
```

### Query

1. Non-multi Agent Use Case

(For non-Mistral API usage)
```
python examples/using_llm_api_as_llm+ollama_embedding.py \
    --run_query \
    --working_directory <Path to store the GraphRAG Backend> \
    --query_input_path <Path to the input txt file that contains the question> \
    --query_output_path <Path to the output file tht outputs the answer>
```
(For Mistral API usage)
```
python examples/using_mistral_api_as_llm+ollama_embedding.py \
    --run_query \
    --working_directory <Path to store the GraphRAG Backend> \
    --query_input_path <Path to the input txt file that contains the question> \
    --query_output_path <Path to the output file tht outputs the answer>
```

2. Multi Agent Use Case
```
python examples/multi_agent_graphrag_parser.py \
    --work_directory <Path to store the GraphRAG Backend> \
    --query_input_path <Path to the input txt file that contains the question> \
    --query_output_path <Path to the output file tht outputs the answer> \
    --mistral // only add this line is using Mistral API
```

3. Run comparison between (without GraphRAG, with GraphRAG, and with Multi Agent GraphRAG)
```
python examples/response_comparison.py \
    --work_directory <Path to store the GraphRAG Backend> \
    --query_input_path <Path to the input txt file that contains the question> \
    --query_output_path <Path to the output file tht outputs the answer> \
    --mistral // only add this line if using Mistral API
```
Subdirectories will be created under the specified output path for each method, including `without_rag`, `with_rag` and `with_multiagent_rag`.
An extra subdirectories 'final_comparison' will store JSON files that contain LLM's evaluation on the responses from each method on three aspects, namely: Comprehensiveness, Diversity and Empowerment.