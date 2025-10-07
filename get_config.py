from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load generation (decoding) config
gen_config = GenerationConfig.from_pretrained(model_name)

# Print all defaults
print(gen_config.to_dict())

# Access individual decode settings
print("Max new tokens:", gen_config.max_new_tokens)
print("Temperature:", gen_config.temperature)
print("Top-k:", gen_config.top_k)
print("Top-p:", gen_config.top_p)
