def print_all_lines_excluding(filepath, label, exclude_lines):
    print(label)
    try:
        with open(filepath, "r", encoding="utf8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines, start=1):
                if idx not in exclude_lines:
                    print(f"Line {idx}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    print("\n")  # Add an empty line after each section

print("\n--- Tokenization Summary ---")

# 1) Hugging Face BERT model Tokenization
filepath1 = r"C:\Users\POO\hf_tokenized.jsonl"
print_all_lines_excluding(filepath1,
    "Hugging Face BERT model Tokenization from https://huggingface.co/tugstugi/bert-base-mongolian-cased/tree/main:",
    [75, 76, 77, 78])

# 2) Custom regex/ NLTK tokenization
filepath2 = r"C:\Users\POO\nltk_tokenized.jsonl"
print_all_lines_excluding(filepath2,
    "Custom regex/ NLTK tokenization:",
    [75, 76, 77, 78])

# 3) Morphological Tokenization
filepath3 = r"C:\Users\POO\morph_tokenized.jsonl"
#print_all_lines_excluding(filepath3,
    #"Morphological Tokenization:",
    #[75, 76, 77, 78])

# 4) Sentence Piece Model Tokenization
filepath4 = r"C:\Users\POO\spm_test.jsonl"
print_all_lines_excluding(filepath4,
    "Sentence piece Model Tokenization from https://github.com/tugstugi/mongolian-bert/blob/master/sentencepiece/mn_cased.model:",
    [75, 76, 77, 78])

print("All tokenizations complete!")