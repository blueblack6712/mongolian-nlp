import json
import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report

# ----- Configuration for BERT -----
BERT_MODEL_PATH = r"C:\Users\POO\Documents\BIT22-26\22-26ArtificialIntelligence\24-25 Second Semester\大数据处理与技术\main src\bert-base-multiligual-cased"  # update this path if needed
DATASET_PATH = r"C:\Users\POO\pos_tagged_dataset_1.jsonl"  # your JSONL file path

# Define a PyTorch Dataset for POS tagging fine-tuning
class POSDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # We expect data to contain "tokens" and "pos_tags"
                self.samples.append(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Build label mapping from dataset
        labels = set()
        for sample in self.samples:
            for label in sample["pos_tags"]:
                labels.add(label)
        self.label_list = sorted(list(labels))
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        pos_tags = sample["pos_tags"]
        # We join the tokens into a sentence (space separated)
        sentence = " ".join(tokens)
        encoding = self.tokenizer(sentence,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_length,
                                  return_tensors="pt")
        # Align labels with tokens:
        # Use tokenizer.word_ids() to map sub-tokens back to words.
        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != current_word:
                current_word = word_id
                # If out-of-range (should not happen if dataset is consistent), default to 0.
                labels.append(self.label2id.get(pos_tags[word_id], 0))
            else:
                labels.append(-100)
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding

# Load local tokenizer and model for BERT (token classification)
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
num_labels =  len(POSDataset(DATASET_PATH, tokenizer).label_list)
model = BertForTokenClassification.from_pretrained(BERT_MODEL_PATH, local_files_only=True, num_labels=num_labels)

# Create dataset and split into train/validation (80/20 split)
full_dataset = POSDataset(DATASET_PATH, tokenizer)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Define compute_metrics function using seqeval
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    true_preds = []
    for pred, label in zip(predictions, labels):
        curr_true = []
        curr_pred = []
        for p_item, l_item in zip(pred, label):
            if l_item != -100:
                curr_true.append(full_dataset.label_list[l_item])
                curr_pred.append(full_dataset.label_list[p_item])
        true_labels.append(curr_true)
        true_preds.append(curr_pred)
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {"precision": report["weighted avg"]["precision"], 
            "recall": report["weighted avg"]["recall"], 
            "f1": report["weighted avg"]["f1-score"]}

# Set up training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("BERT Fine-tuning started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
trainer.train()
print("BERT Fine-tuning finished at:", time.strftime("%Y-%m-%d %H:%M:%S"))

trainer.save_model("./bert_finetuned")
