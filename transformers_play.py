import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_path = "./distilgpt2-wikitext2-m1-final"
print(f"Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to MPS if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
print(f"Running on device: {device}")

# Set model to evaluation mode
model.eval()


def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, top_k=50):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Example prompts
    prompts = [
        "The history of science",
        "In the early morning",
        "The president announced",
    ]
    
    print("\n" + "=" * 60)
    print("Text Generation with Fine-tuned DistilGPT-2")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        generated = generate_text(prompt, max_new_tokens=50)
        print(f"Generated: {generated}")
        print()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        user_prompt = input("\nEnter prompt: ").strip()
        if user_prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_prompt:
            continue
        
        generated = generate_text(user_prompt, max_new_tokens=100)
        print(f"\nGenerated:\n{generated}")

