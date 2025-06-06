import json
import os
import time
import gc
import torch
import random
from torch.utils.data import Dataset, Subset
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import dispatch_model
import bitsandbytes as bnb  # For 4-bit quantization

# ====== CONFIGURATION ======
LLAMA_MODEL_PATH = "/dev/shm/Llama-2-3b-hf/"
DATASET_PATH_LLAMA = "/dev/shm/train.jsonl"
USE_LORA = True  # Enable LoRA to reduce memory usage
USE_4BIT = True  # Use 4-bit quantization for efficient training
DOWNSAMPLE_RATIO = 0.3  # Adjust dataset size (e.g., 0.3 = 30% of full data)

# ====== MEMORY MANAGEMENT FUNCTIONS ======
def clear_cuda_memory():
    """Clears unused GPU memory to prevent OutOfMemory errors."""
    gc.collect()
    torch.cuda.empty_cache()

# ====== DEFINE DATASET ======
class LlamaPOSDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                prompt = data["original_text"]
                completion = " ".join(data["pos_tags"])
                self.samples.append({"prompt": prompt, "completion": completion})
        
        # Downsample dataset to reduce training time
        if 0 < DOWNSAMPLE_RATIO < 1:
            self.samples = random.sample(self.samples, int(len(self.samples) * DOWNSAMPLE_RATIO))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["prompt"] + " " + sample["completion"]
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

# ====== LOAD TOKENIZER & MODEL ======
try:
    tokenizer_llama = LlamaTokenizer.from_pretrained(
        LLAMA_MODEL_PATH, 
        tokenizer_file=os.path.join(LLAMA_MODEL_PATH, "tokenizer.model"),
        legacy=False,
        local_files_only=True
    )

    # **Fix**: Set padding token to eos token
    tokenizer_llama.pad_token = tokenizer_llama.eos_token  # Use eos_token as pad_token

    model_llama = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True).cuda()

    print("Model and Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Model or Tokenizer could not be loaded.")

# **Fix**: Prepare 4-bit model for LoRA
if USE_4BIT:
    model_llama = prepare_model_for_kbit_training(model_llama)

# Apply LoRA for low-rank adaptation
if USE_LORA:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=8,  # Low-rank dimension (adjust as needed)
        lora_alpha=16,  
        lora_dropout=0.05
    )
    model_llama = get_peft_model(model_llama, peft_config)

# ====== CREATE TRAIN & VALIDATION DATASETS ======
dataset_llama = LlamaPOSDataset(DATASET_PATH_LLAMA, tokenizer_llama)
dataset_size = len(dataset_llama)
split_point = int(0.8 * dataset_size)
train_dataset_llama = Subset(dataset_llama, list(range(split_point)))
val_dataset_llama = Subset(dataset_llama, list(range(split_point, dataset_size)))

# ====== TRAINING ARGUMENTS ======
training_args_llama = TrainingArguments(
    output_dir="./llama_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Increase if memory allows
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Adjust to fit memory
    eval_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    logging_dir="./logs_llama",
    fp16=True,  # Enable mixed precision training
    optim="adamw_bnb_8bit" if USE_4BIT else "adamw_torch_fused",  # Use optimized optimizer
    torch_compile=True,
    save_steps=500,  # Save the model every 500 steps
)

# ====== TRAINER SETUP ======
trainer_llama = Trainer(
    model=model_llama,
    args=training_args_llama,
    train_dataset=train_dataset_llama,
    eval_dataset=val_dataset_llama,
    tokenizer=tokenizer_llama
)

# ====== TRAINING LOOP WITH MEMORY MANAGEMENT ======
print("LLaMA Fine-tuning started at:", time.strftime("%Y-%m-%d %H:%M:%S"))

try:
    trainer_llama.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("CUDA Out of Memory! Reducing batch size...")
        clear_cuda_memory()

        # Retry with lower batch size
        training_args_llama.per_device_train_batch_size = max(1, training_args_llama.per_device_train_batch_size // 2)
        training_args_llama.gradient_accumulation_steps *= 2  # Compensate for smaller batches

        trainer_llama = Trainer(
            model=model_llama,
            args=training_args_llama,
            train_dataset=train_dataset_llama,
            eval_dataset=val_dataset_llama,
            tokenizer=tokenizer_llama
        )

        trainer_llama.train()

print("LLaMA Fine-tuning finished at:", time.strftime("%Y-%m-%d %H:%M:%S"))

# ====== SAVE FINAL MODEL ======
trainer_llama.save_model("./llama_finetuned")
print("Final model saved.")


