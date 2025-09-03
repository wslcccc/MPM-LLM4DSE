# MPM-LLM4DSE: Reaching the Pareto Frontier in HLS with Multimodal Learning and LLM-Driven Exploration  

## Content
- [About the project](#jump1)
- [Project File Tree](#jump2)
- [Required environment](#jump3)

## <span id="jump1">About the project</span>

This project presents an end-to-end framework utilizing deep learning models as substitutes for HLS tools to obtain QoR results through predictive modeling. Concurrently, it incorporates our proposed LLM4DSE framework to leverage LLMs' semantic comprehension capabilities for intelligent design space exploration, employing trained predictive models for QoR estimation to identify Pareto-optimal solutions.

![The framework of MPM-LLM4DSE](framework.jpeg)

### Contribution
we propose MPM-LLMDSE, an automated framework featuring:
- A Multimodal Dataset：Constructs paired data instances combining CDFG structural features with semantic embeddings extracted from pragma-augmented source code using a pre-trained language model (LM).
- A Multimodal Prediction Model (MPM)：Proposes a hybrid architecture leveraging language models and GNNs to extract complementary features from source code and CDFG respectively, dynamically fused via multi-head attention mechanisms.
- An LLM-Driven DSE Engine (LLM4DSE)：Introduces a sophisticated prompt engineering methodology (PEODSE) that incorporates pragma impact analysis and high-quality examples to guide LLMs in generating high-quality design configurations iteratively.

## <span id="jump2">Project File Tree</span>
```
|--MPM-LLM4DSE
|--dse_database                    # database, graphs, codes to generate/analyze them
|--ECoGNN                          # components of ECoGNN
|   +--action_gumbel_layer.py
|   +--layers.py
|   +--model_parse.py
|--LLM4DSE                         # the implementation details of LLM4DSE and the comparison of prompt word methods
|   +--dse.py
|   +--LLM.py
|   +--prompt.py
|--Save the model parameters       # save model weights and generate the files required for CDFGs
|--src                             # the source codes for defining and training the model and running the DSE
|   +--config.py
|   +--config_ds.py
|   +--main.py
|   +--model.py
|   +--nn_att.py
|   +--parallel_run_tool_dse.py
|   +--parameter.py
|   +--programl_data.py
|   +--result.py
|   +--saver.py
|   +--train.py
|   +--utils.py
```

## <span id="jump3">Required environment</span>
- os Linux
- python 3.9.22
- torch 1.12.1+cu118
- torch_geometric 2.2.0
- openai 0.28.1
- numpy 1.23.5
- langchain 0.1.16
