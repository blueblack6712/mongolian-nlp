import json
import random
import os

# Update these paths accordingly
input_dataset_path = "/dev/shm/pos_tagged_dataset_new.jsonl"  # Your full dataset (1GB)
train_output_path = "/dev/shm/train.jsonl"
valid_output_path = "/dev/shm/validation.jsonl"
test_output_path  = "/dev/shm/test.jsonl"

# Set split ratios (80% train, 10% validation, 10% test)
train_ratio = 0.8
valid_ratio = 0.1
# test_ratio = 0.1  (implicitly, remaining)

# Remove output files if they exist
for path in [train_output_path, valid_output_path, test_output_path]:
    if os.path.exists(path):
        os.remove(path)

print("Splitting dataset...")
with open(input_dataset_path, "r", encoding="utf-8") as fin, \
     open(train_output_path, "w", encoding="utf-8") as ftrain, \
     open(valid_output_path, "w", encoding="utf-8") as fvalid, \
     open(test_output_path, "w", encoding="utf-8") as ftest:
    
    for line in fin:
        if not line.strip():
            continue
        rnd = random.random()
        if rnd < train_ratio:
            ftrain.write(line)
        elif rnd < train_ratio + valid_ratio:
            fvalid.write(line)
        else:
            ftest.write(line)

print("Dataset splitting complete!")
print(f"Train: {train_output_path}\nValidation: {valid_output_path}\nTest: {test_output_path}")
