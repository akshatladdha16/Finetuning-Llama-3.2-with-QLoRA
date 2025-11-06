---
base_model: meta-llama/Llama-3.2-3B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.2-3B
- lora
- transformers
language:
- en
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->
This is a PEFT (Parameter Efficient Fine-Tuning) adapter trained on chemistry educational content using QLoRA (Quantized Low-Rank Adaptation) technique. The adapter is designed to enhance Llama-3.2-3B's capabilities in answering chemistry-related questions.



## Model Details
- **Base Model**: meta-llama/Llama-3.2-3B
- **Training Technique**: QLoRA (4-bit quantization)
- **Domain**: Chemistry Education
- **Language**: English
- **License**: Same as base model

### Model Description

This model is a QLoRA fine-tuned version of Meta-Llama-3.2-3B specifically optimized for chemistry question-answering tasks. The adapter layers were trained on a diverse chemistry dataset containing 4.4k+ educational Q&A pairs covering fundamental to advanced chemistry concepts.


## Use Cases
- Answering chemistry concepts and definitions
- Explaining chemical processes and reactions
- Solving basic chemistry problems
- Providing chemistry educational content



## Example Usage
You can infer the adapter layers with base model here : [https://colab.research.google.com/drive/16N_lnLKieJjMunvIXb59LtGavifx96nx#scrollTo=nd3kQhZbm2z9]

## Training Setup
- **Training Type**: QLoRA fine-tuning
- **Hardware**: 4GB VRAM GPU optimization
- **Quantization**: 4-bit (NF4 format)
- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 32
  - Target Modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  - Dropout: 0.05

### Training Data

The model was fine-tuned on a curated dataset of chemistry educational content, focusing on:
- Basic chemistry concepts
- Chemical processes and reactions
- Problem-solving examples
- NCERT chemistry curriculum
  
[https://huggingface.co/datasets/KadamParth/NCERT_Chemistry_12th]


## Training Setup

### Preprocessing
- **Data Format**: Chat-format JSONL with messages/roles structure
- **Tokenization**:
  - Max Length: 512 tokens
  - Padding: Right-side padding with EOS token
  - Special Tokens: Added conversation markers (User:, Assistant:)
- **Prompt Template**:
  ```
  ### Conversation:
  User: {chemistry_question}
  
  Assistant: {response}
  ```

### Training Procedure
- **Hardware**: Single GPU with 4GB VRAM optimization
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Base Quantization**: 4-bit NF4 format with double quantization
- **Memory Optimizations**:
  - Gradient Checkpointing: Enabled
  - Mixed Precision (fp16)
  - 8-bit Adam optimizer
  - Gradient accumulation
- **Training Progress**:
  - Evaluation every 50 steps
  - Model checkpoints every 200 steps
  - TensorBoard logging enabled

### Hyperparameters
- **Training Configuration**:
  - Epochs: 2
  - Batch Size: 1 (per device)
  - Gradient Accumulation Steps: 16
  - Effective Batch Size: 16
  - Learning Rate: 1e-4
  - Warmup Steps: 50

- **LoRA Settings**:
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: 
    - Attention: q_proj, k_proj, v_proj, o_proj
    - FFN: gate_proj, up_proj, down_proj
  - Dropout: 0.05
  - Bias: none
  - Task Type: CAUSAL_LM


#### Hardware
NVIDIA RTX 3050 with 4GB VRAM

## Limitations
- Limited to chemistry domain knowledge
- Performance depends on base model capabilities
- May require 4GB+ VRAM for inference with quantization
- Responses should be verified for accuracy

## Citation
If you use this model, please cite:
```bibtex
@misc{llama-chemistry-adapter,
  author = {Akshat Rai Laddha},
  title = {Chemistry QLoRA Adapter for Llama-3.2-3B},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
}
```


### Framework versions

- PEFT 0.17.1