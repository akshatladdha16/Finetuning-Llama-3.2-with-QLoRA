import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, #for text tokenization and detokenization -> by HF 
    AutoModelForCausalLM,  #loads the base model for next word pred 
    TrainingArguments,  #define hyperparams and other training settings by HF
    Trainer, #Training and eval loops 
    BitsAndBytesConfig #for quantization settings (4-bit here) by HF -> saves GPU memory, large models get fit into smaller vram
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training #parameter efficient fine tuning (only train small subsets of model)
#get_peft_model -> applies lora to base model , prepare-> get quantized model ready for fientuning without precision errors
from transformers import DataCollatorForLanguageModeling #dynamic batching and padding during training
import os

# Set device to GPU and enable memory optimizations
torch.cuda.set_device(0)  # Force GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Load the dataset
def load_ncert_chemistry_data():
    """Load dataset for fine-tuning.

    This project requires a local JSONL file `chemistry_finetune.jsonl` in the
    project root using the chat-format (each line contains a `messages` list
    with role/content dicts). There is no fallback to external Hugging Face
    datasets — the local file must be present.
    """
    local_path = os.path.join(os.path.dirname(__file__), "chemistry_finetune.jsonl") #already preprocessed into chat format for easy finetuning
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Required local dataset not found at {local_path}. Please add the chat-format JSONL file."
        )
    print(f"Loading local JSONL dataset at: {local_path}")
    dataset = load_dataset("json", data_files=local_path)
    return dataset

# Initialize tokenizer for Llama 3.2
def initialize_tokenizer():
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name) #auto selects the tokenizer for the model
    
#     # Llama tokenizer settings
#     Some models like LLaMA don’t define a dedicated padding token.
# Padding tokens are needed when batching sequences of different lengths (so shorter ones align with the longest).avoids runtime errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" #pad at end of the sequnece
    
    return tokenizer

# Preprocess function for Llama
def preprocess_function(examples, tokenizer, max_length=512):
    """Create tokenized inputs from a batch of chat-format examples.

    This function assumes the dataset is chat-format JSONL where each example
    contains a `messages` list of role/content dicts. It will raise an error
    if the expected `messages` key is missing — no fallback to other column
    names is performed.

    Args:
        examples: Dataset examples containing messages field
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length (default: 512)
    """
    if "messages" not in examples:
        raise ValueError("Dataset must be chat-format JSONL with a 'messages' field.")

    texts = []
    source_fields = []

    # examples['messages'] is a list of message-lists (batched)
    for msg_list in examples["messages"]:
        user_parts = []
        assistant_parts = []
        if isinstance(msg_list, (list, tuple)):
            for m in msg_list:
                # m expected to be a dict like {'role': 'user', 'content': '...'}
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                else:
                    # fallback if messages are plain strings
                    role = "user"
                    content = str(m)
                if role == "assistant":
                    assistant_parts.append(str(content).strip())
                else:
                    user_parts.append(str(content).strip())

        user_text = "\n".join([p for p in user_parts if p])
        assistant_text = "\n".join([p for p in assistant_parts if p])

        if user_text and assistant_text:
            formatted = f"### Conversation:\nUser: {user_text}\n\nAssistant: {assistant_text}"
        elif user_text:
            formatted = f"### Conversation:\nUser: {user_text}\n\nAssistant:"
        else:
            formatted = assistant_text or ""

        texts.append(formatted)
        source_fields.append("messages")

    # Tokenize the generated texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,  # Enable padding(model gives output of fixed tokens but this may vary from generated output tokoens, so for model to process consistent tokens we use padding)
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True,
    )

    # Labels: copy of input ids (can be changed to mask prompts if desired) so that loss is only comuted for the generated part
    tokenized["labels"] = [ids.copy() for ids in tokenized.get("input_ids", [])]
    # Remove source_fields as it's not needed for training
    return tokenized

def setup_qlora_config():
    """Configure QLoRA for Llama 3.2 3B"""
    lora_config = LoraConfig(
        r=16,  # LoRA rank , higher r makes better training but more GPu usgae
        lora_alpha=32,  # LoRA alpha : scaling factor-> means controls how much Lora layers contribute to final output compared to base model
        target_modules=[ #which layer in the model Lora will modify
            "q_proj", "k_proj", "v_proj", "o_proj", #query, key , value and output projection layers in attention mechanism
            "gate_proj", "up_proj", "down_proj", #feed forward MLP layers
        ],
        lora_dropout=0.05, #helps prevent overfitting during training by randomly deactivating some Lora connections (0 to 0.1)
        bias="none", #low cost params-> per output unit scalars (none so that there are not trained)
        task_type="CAUSAL_LM", 
    )
    return lora_config

def load_model_with_qlora():
    """Load Llama 3.2 3B with QLoRA for 4GB VRAM"""
    model_name = "meta-llama/Llama-3.2-3B" 
    
    # Quantization configuration for 4GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, #backward and forward propogation in half precision(good speed accuracxy balancu)
    ) #quantization configuration for 4-bit quantization 
    
    print("Loading Llama 3.2 3B with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model) #only for transformer models 
    
    # Apply LoRA
    lora_config = setup_qlora_config()
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    return model

def print_trainable_parameters(model):
    """Print the number of trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )

class ChemistryLlamaTrainer:
    def __init__(self):
        #only initialized here
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup(self):
        """Setup all components"""
        print("Loading dataset...")
        self.dataset = load_ncert_chemistry_data()
        
        print("Initializing tokenizer...")
        self.tokenizer = initialize_tokenizer() #for llama models
        
        print("Loading Llama 3.2 3B with QLoRA...")
        self.model = load_model_with_qlora() #final model with lora applied
        
        print("Preprocessing data...")
        self._preprocess_data()
        
        print_trainable_parameters(self.model)
        check_gpu_memory()
    
    def _preprocess_data(self):
        """Preprocess the dataset"""
        # Use the first available split's column names for remove_columns
        first_split = list(self.dataset.keys())[0]
        tokenized_dataset = self.dataset.map(
            lambda x: preprocess_function(x, self.tokenizer),
            batched=True,
            remove_columns=self.dataset[first_split].column_names,
        )
        
        # Split into train and validation
        if 'train' in tokenized_dataset:
            train_val_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
            self.train_dataset = train_val_split['train']
            self.eval_dataset = train_val_split['test']
        else:
            # If no 'train' split, use the first available split
            first_key = list(tokenized_dataset.keys())[0]
            train_val_split = tokenized_dataset[first_key].train_test_split(test_size=0.1)
            self.train_dataset = train_val_split['train']
            self.eval_dataset = train_val_split['test']
    
    def train(self):
        """Start training with GPU optimization"""
        # Training arguments optimized for 4GB VRAM
        training_args = TrainingArguments(
            output_dir="./llama-chemistry-model",
            overwrite_output_dir=True,
            num_train_epochs=2,   #kept it low to avoid overfitting of adapter layers
            per_device_train_batch_size=1,  # Essential for 4GB VRAM : no. of sample processign per gpu
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Effective batch size = 1 * 16 = 16
            warmup_steps=50,
            learning_rate=1e-4,  # Slightly lower for stable training
            fp16=True,  # Mixed precision
            bf16=False,  # Keep as False for RTX 3050
            logging_steps=10,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=50,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,  # Save memory
            optim="paged_adamw_8bit",  # Optimized optimizer for 8-bit from bitsandBytes
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling( #task is to process batches for model training. comvert dataset into single padded batch that model can accept
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("Starting training...")
        check_gpu_memory()
        
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained("./llama-chemistry-model")
        
        print("Training completed!")
        check_gpu_memory()
        
        return trainer
    
def generate_response(model, tokenizer, prompt, max_length=300):
    """Generate response using the fine-tuned Llama model"""
    model.eval()
    
    # Format prompt
    formatted_prompt = f"### Chemistry Question: {prompt}\n\n### Answer:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "### Answer:" in response:
        response = response.split("### Answer:")[-1].strip()
    
    return response

def test_model():
    """Test the fine-tuned model"""
    from peft import PeftModel, PeftConfig
    
    try:
        # Load fine-tuned model
        config = PeftConfig.from_pretrained("./llama-chemistry-model")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(base_model, "./llama-chemistry-model")
        tokenizer = AutoTokenizer.from_pretrained("./llama-chemistry-model")
        
        # Test prompts
        test_prompts = [
            "What is the definition of molarity?",
            "Explain the concept of chemical equilibrium.",
            "Describe the process of electrolysis.",
            "What are coordination compounds?",
            "How do you calculate pH of a solution?",
        ]
        
        print("Testing the fine-tuned Llama 3.2 3B model:")
        print("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}:")
            print(f"Prompt: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error loading model for testing: {e}")
        print("You may need to train the model first.")
def main():
    """Main execution function"""
    print("Llama 3.2 3B NCERT Chemistry Fine-tuning Project")
    print("=" * 60)
    
    # Verify GPU
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set GPU device explicitly
    torch.cuda.set_device(0)
    
    # Initialize trainer
    trainer = ChemistryLlamaTrainer()
    
    try:
        # Setup components
        trainer.setup()
        
        # Start training
        trained_trainer = trainer.train()
        
        print("\nTraining completed successfully!")
        
        # Test the model
        print("\nTesting the model...")
        test_model()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()