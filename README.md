# Chemistry QLoRA Adapter for Llama-3.2-3B

This is a PEFT (Parameter Efficient Fine-Tuning) adapter trained on chemistry educational content using QLoRA (Quantized Low-Rank Adaptation) technique. The adapter is designed to enhance Llama-3.2-3B's capabilities in answering chemistry-related questions.

## Model Details
- **Base Model**: meta-llama/Llama-3.2-3B
- **Training Technique**: QLoRA (4-bit quantization)
- **Domain**: Chemistry Education
- **Language**: English
- **License**: Same as base model

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


## Use Cases
- Answering chemistry concepts and definitions
- Explaining chemical processes and reactions
- Solving basic chemistry problems
- Providing chemistry educational content

## Example Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load adapter and tokenizer
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/llama-chemistry-adapter")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/llama-chemistry-adapter")

# Example inference
def generate_response(prompt, max_length=300):
    inputs = tokenizer(f"### Chemistry Question: {prompt}\n\n### Answer:", 
                      return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test examples
questions = [
    "What is the definition of molarity?",
    "Explain chemical equilibrium",
    "How do you calculate pH?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {generate_response(q)}\n")
```

## Training Data
The model was fine-tuned on a curated dataset of chemistry educational content, focusing on:
- Basic chemistry concepts
- Chemical processes and reactions
- Problem-solving examples
- NCERT chemistry curriculum

## Limitations
- Limited to chemistry domain knowledge
- Performance depends on base model capabilities
- May require 4GB+ VRAM for inference with quantization
- Responses should be verified for accuracy

## Citation
If you use this model, please cite:
```bibtex
@misc{llama-chemistry-adapter,
  author = {Your Name},
  title = {Chemistry QLoRA Adapter for Llama-3.2-3B},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
}
```