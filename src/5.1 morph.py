import re
import json
from collections import OrderedDict
import time

# File paths
INPUT_JSONL = r"C:\Users\POO\Documents\BIT22-26\22-26ArtificialIntelligence\24-25 Second Semester\大数据处理与技术\main src\test_cleaned.jsonl"
OUTPUT_JSONL = r"C:\Users\POO\morph_tokenized.jsonl"

class MongolianTokenizer:
    def __init__(self):
        # Ordered dictionary of Mongolian suffixes (longest first)
        self.suffix_rules = OrderedDict([
            ('тэй', 'тэй'), ('ийг', 'ийг'), ('ын', 'ын'), 
            ('ийн', 'ийн'), ('д', 'д'), ('т', 'т'),
            ('аас', 'аас'), ('ээс', 'ээс'), ('ий', 'ий'),
            ('ыг', 'ыг'), ('руу', 'руу'), ('луу', 'луу'),
            ('нээс', 'нээс'), ('нээр', 'нээр'), ('чих', 'чих'),
            ('жуул', 'жуул'), ('чхуу', 'чхуу'), ('гуй', 'гуй')
        ])
        
        # Base word pattern with Mongolian-specific characters
        self.base_pattern = re.compile(
            r"[\w\u0400-\u04FF’]+(?:[-’][\w\u0400-\u04FF’]+)*|[\.,!?;]"
        )

    def tokenize(self, text):
        base_tokens = self.base_pattern.findall(text)
        processed = []
        
        for token in base_tokens:
            if token.isalpha():
                # Try longest suffixes first
                for suf in self.suffix_rules:
                    if token.endswith(suf):
                        root = token[:-len(suf)]
                        if len(root) > 1:  # Prevent over-splitting
                            processed.extend([root, suf])
                            break
                else:
                    processed.append(token)
            else:
                processed.append(token)
        return processed

def morphological_tokenization():
    print("Initializing Mongolian tokenizer...")
    tokenizer = MongolianTokenizer()
    
    print("Processing morphological tokenization...")
    start_time = time.time()
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            data = json.loads(line)
            text = data["text"]
            
            tokens = tokenizer.tokenize(text)
            
            f_out.write(json.dumps({
                "original_text": text,
                "morph_tokens": tokens
            }, ensure_ascii=False) + "\n")
    
    print(f"Completed in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    morphological_tokenization()