import json
import stanza
from multiprocessing import Pool, cpu_count
import psutil
import re
from pathlib import Path
import os

# Configuration
STANZA_RESOURCES_DIR = r"C:\stanza_resources"
INPUT_DIR = r"C:\Users\POO\Desktop\test"
OUTPUT_DIR = r"C:\Users\POO\tokenized_output"

MAX_LINE_LENGTH = 5000
BATCH_SIZE = 1000
LOG_INTERVAL = 10000

def setup_stanza():
    """One-time resource setup"""
    resource_dir = Path(STANZA_RESOURCES_DIR)
    resource_dir.mkdir(exist_ok=True)
    
    if not (resource_dir / "mn").exists():
        print("Downloading Mongolian resources...")
        stanza.download("mn", model_dir=str(resource_dir), logging_level="INFO")

def init_stanza():
    global nlp
    try:
        nlp = stanza.Pipeline(
            lang="mn",
            dir=STANZA_RESOURCES_DIR,
            processors="tokenize,pos",
            use_gpu=True,
            tokenize_batch_size=100,
            pos_batch_size=100
        )
    except Exception as e:
        print(f"Stanza initialization failed: {str(e)}")
        raise

def clean_text(text):
    """Optimized text cleaning"""
    text = re.sub(r'\b\d{4}-\d{1,2}-\d{1,2}\b', '', text)  # Remove dates
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    text = re.sub(r'(Үнийн санал авах|ҮР ДҮН:|warc-\S+)', '', text)  # Remove patterns
    return text.strip()

def process_batch(batch):
    results = []
    for line in batch:
        try:
            data = json.loads(line)
            text = clean_text(data.get("content", ""))
            if not text or len(text) > MAX_LINE_LENGTH:
                continue

            doc = nlp(text)
            results.append(json.dumps({
                "text": text,
                "tokens": [word.text for sent in doc.sentences for word in sent.words],
                "pos_tags": [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
            }, ensure_ascii=False))
            
        except Exception as e:
            print(f"Error processing line: {str(e)}")
    return results

def chunk_reader(file_path):
    """Memory-efficient batch reader"""
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            batch = [line.strip() for _ in range(BATCH_SIZE) if (line := f.readline())]
            if not batch:
                break
            yield batch

def process_file(file_pair):
    input_file, output_file = file_pair
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        processed = 0
        for batch in chunk_reader(input_file):
            for result in process_batch(batch):
                f_out.write(result + "\n")
            processed += len(batch)
            if processed % LOG_INTERVAL == 0:
                mem = psutil.virtual_memory().percent
                print(f"Processed {processed} lines | Memory: {mem}%")

def run_pipeline():
    # Create directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_files = list(Path(INPUT_DIR).glob("*.jsonl"))
    if not input_files:
        print(f"No JSONL files found in {INPUT_DIR}")
        return

    file_pairs = [(str(f), str(Path(OUTPUT_DIR) / f"{f.stem}_processed.jsonl")) for f in input_files]
    
    # Use 75% of available cores
    workers = max(1, int(cpu_count() * 0.75))
    
    with Pool(processes=workers, initializer=init_stanza) as pool:
        pool.map(process_file, file_pairs)

if __name__ == "__main__":
    # Verify and setup resources
    print(f"Stanza resources directory: {STANZA_RESOURCES_DIR}")
    setup_stanza()
    
    # Run pipeline
    run_pipeline()
    print("Processing completed successfully!")