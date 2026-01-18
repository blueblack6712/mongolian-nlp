import json  # Use Python's built-in JSON module
import re
import time  # For timing the execution
from unicodedata import normalize
from multiprocessing import Pool, freeze_support
from functools import partial
from tqdm import tqdm  # For progress bar in Jupyter Notebook

# Precompile all regex patterns for significant speedup
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
DOC_ARTIFACTS = [
    (re.compile(r"\nҮнийн санал авах\n"), ""),
    (re.compile(r"ҮР ДҮН: \d+-\d+ ХҮРТЭЛ / НИЙТ: \d+"), ""),
    (re.compile(r"Сар\d{2}|Өдөр\d{2}"), ""),
]
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\d{4} оны"),
    re.compile(r"\b\d{4}\b"),
]
NUMBER_PATTERNS = [
    re.compile(r"\b\d{5,}\b"),
    re.compile(r"\d+[\d\s]+\d+"),
]
SPECIAL_PATTERNS = [
    (re.compile(r"Шүүх\n+"), ""),
    (re.compile(r"warc-.*?}"), ""),
    (re.compile(r'"quality_warnings":\[.*?\]'), ""),
]
FOOTER_PATTERNS = [
    (re.compile(r"Нийтлэлч\n.*?Б\. Ундрам", re.DOTALL), ""),
    (re.compile(r"Төрөл\nМэдээ\nВидео.*?Ангиллууд", re.DOTALL), ""),
]
CLEAN_CHARS = re.compile(r"[^а-яөүёА-ЯӨҮЁ.,!?;:()\"'\s]")
WHITESPACE_PATTERNS = [
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"[ \t]+\n"), "\n"),
]

def clean_mongolian_text(text):
    """Optimized text cleaning with precompiled patterns"""
    text = normalize("NFC", text)
    
    # Apply all patterns in sequence
    text = URL_PATTERN.sub("", text)
    
    for pattern, replacement in DOC_ARTIFACTS:
        text = pattern.sub(replacement, text)
        
    for pattern in DATE_PATTERNS:
        text = pattern.sub("", text)
        
    for pattern in NUMBER_PATTERNS:
        text = pattern.sub("", text)
        
    for pattern, replacement in SPECIAL_PATTERNS:
        text = pattern.sub(replacement, text)
        
    for pattern, replacement in FOOTER_PATTERNS:
        text = pattern.sub(replacement, text)
        
    text = CLEAN_CHARS.sub("", text)
    
    for pattern, replacement in WHITESPACE_PATTERNS:
        text = pattern.sub(replacement, text)
        
    return text.strip()

def process_line(line):
    """Optimized line processor with reduced checks"""
    try:
        line = line.strip()
        if not line or not line.startswith("{"):
            return None
            
        data = json.loads(line)
        if "content" not in data:
            return None
            
        cleaned = clean_mongolian_text(data["content"])
        return json.dumps({"text": cleaned}, ensure_ascii=False) if cleaned.strip() else None
    except Exception:
        return None

def clean_jsonl_parallel(input_file, output_file):
    """Optimized parallel processing with better resource usage"""
    freeze_support()
    
    # Start timing
    start_time = time.time()
    
    # Use 15 workers (leaves 1 core free for system)
    with Pool(15) as pool:
        # Use larger buffers for I/O (16MB buffer)
        with open(input_file, "r", encoding="utf-8", buffering=16*1024*1024) as f_in:
            # Get total number of lines for progress bar
            total_lines = sum(1 for _ in f_in)
            f_in.seek(0)  # Reset file pointer
            
            # Use unordered imap for faster processing
            results = pool.imap_unordered(process_line, f_in, chunksize=5000)
            
            # Open output file
            with open(output_file, "w", encoding="utf-8", buffering=16*1024*1024) as f_out:
                # Use tqdm for progress bar
                for result in tqdm(results, total=total_lines, desc="Processing", unit="lines"):
                    if result:
                        f_out.write(result + "\n")
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"✅ Successfully saved cleaned data to {output_path}")
    print(f" Total time taken: {total_time:.2f} seconds")

if __name__ == '__main__':
    input_path = r"/dev/shm/merged_mongolian_dataset.jsonl"
    output_path = r"/dev/shm/cleaned_mongolian_dataset.jsonl"
    
    # Skip inspection for faster execution
    clean_jsonl_parallel(input_path, output_path)