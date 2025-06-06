import os
from transformers import LlamaForCausalLM, LlamaTokenizer

# Path to your model
LLAMA_MODEL_PATH = "/dev/shm/Llama-2-3b-hf/"

# List the files in the model path to ensure the correct files are present
print("Model files in directory:")
for file_name in os.listdir(LLAMA_MODEL_PATH):
    print(file_name)

# Load tokenizer and model
try:
    # Ensure the tokenizer is correctly loaded
    tokenizer_llama = LlamaTokenizer.from_pretrained(
        LLAMA_MODEL_PATH,
        tokenizer_file=os.path.join(LLAMA_MODEL_PATH, "tokenizer.model"),
        legacy=False,
        local_files_only=True
    )

    # Load the model
    model_llama = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True).cuda()

    print("Model and Tokenizer loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
