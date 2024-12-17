# CSCI5120 Project - Multi-agent Graph RAG and Evaluation

Group members: CHUNG Wai Kei, CUI Wenqian, PAN Chengjie,  SEW Kin Hang

Acknowledgement: Part of the code is modified from https://github.com/gusye1234/nano-graphrag and https://github.com/yixuantt/MultiHop-RAG

## Installation

Setup conda environment
```
conda env create --name custom_env_name -f environment.yml
```

Download Ollama from https://github.com/ollama/ollama/releases

Pull new model 
```
./ollama-linux-amd64/bin/ollama pull qwen2 
```

## Usage

Start ollama model
```
./ollama-linux-amd64/bin/ollama serve& 
export no_proxy=localhost,127.0.0.0,127.0.0.1,127.0.1.1 
```

Setup nano-graphrag
```
cd <Path to nano-graphrag>
export LLM_BASE_URL="https://api.deepseek.com" // or other api
export LLM_API_KEY="YOUR-KEY"  
```

Insert documents
```
python examples/using_llm_api_as_llm+ollama_embedding.py \
    --run_insert \
    --documents_path <Path to document folder containing txt files> \
```

Query
```
TODO
```
