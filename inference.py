import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path="./llama-chemistry-model/checkpoint-200"):
    """Load the fine-tuned model"""
    from peft import PeftConfig
    
    # First load the PEFT config to get base model name
    peft_config = PeftConfig.from_pretrained(model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model with 4-bit quantization config
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(
        base_model, 
        model_path,
        is_trainable=False  # Set to inference mode
    )
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=300):
    """Generate response using the fine-tuned model"""
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

def main():
    # Test prompts
    test_prompts = [
       "Describe the spatial arrangement of the cis-isomer of [PtCl2(en)2]2+ that allows it to exhibit optical activity"
    ]
    
    try:
        print("Loading fine-tuned model...")
        model, tokenizer = load_model()
        
        print("\nTesting the fine-tuned model:")
        print("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}:")
            print(f"Question: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            print(f"Answer: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()