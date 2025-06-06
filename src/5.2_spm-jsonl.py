import sentencepiece as spm
import time
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
from datetime import datetime
import psutil
import mmap

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = r'/dev/shm/mn_cased.model'
INPUT_FILE = r'/dev/shm/cleaned_mongolian_dataset.jsonl'
OUTPUT_FILE = r'/dev/shm/spm_mongolian_dataset.jsonl'

# Resource allocation
TOTAL_CORES = cpu_count()
NUM_WORKERS = max(1, int(TOTAL_CORES * 0.75))
CHUNK_SIZE = 50000 if NUM_WORKERS > 32 else 20000
BUFFER_SIZE = 1024 * 1024 * 16
MAX_RETRIES = 3
LOG_INTERVAL = 5

# --------------------------
# Initialize Resources
# --------------------------
sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PATH)

# --------------------------
# Helper Functions
# --------------------------
def get_system_stats():
    mem = psutil.virtual_memory()
    return {
        'cpu_usage': psutil.cpu_percent(),
        'mem_used': mem.used / (1024**3),
        'mem_total': mem.total / (1024**3)
    }

def format_interval(seconds):
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

def tokenize_line(line):
    for attempt in range(MAX_RETRIES):
        try:
            data = json.loads(line.strip())
            text = data.get('text', '')
            tokens = sp.EncodeAsPieces(text)
            return json.dumps({'text': ' '.join(tokens)}, ensure_ascii=False) + '\n'
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(0.1 * attempt)
    return None

# --------------------------
# Processing Functions
# --------------------------
def process_chunk(chunk):
    with Pool(NUM_WORKERS) as pool:
        results = list(pool.imap(tokenize_line, chunk, chunksize=1000))
    return results

# --------------------------
# Main Execution
# --------------------------
def main():
    start_time = time.time()
    stats = {
        'success_lines': 0,
        'error_lines': 0,
        'bytes_written': 0
    }

    # System checks
    if os.path.exists(OUTPUT_FILE):
        raise FileExistsError(f"Output file {OUTPUT_FILE} already exists!")
    
    # Count total lines
    print("Counting total lines...")
    count_start = time.time()
    with open(INPUT_FILE, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        total_lines = 0
        while mm.readline():
            total_lines += 1
        mm.close()
    print(f"Counted {total_lines:,} lines in {time.time()-count_start:.2f}s")

    # Initialize progress bar
    progress = tqdm(total=total_lines, desc="Tokenizing", unit="line")

    with open(INPUT_FILE, 'r+b') as infile, \
         open(OUTPUT_FILE, 'w', buffering=BUFFER_SIZE) as outfile:

        mm = mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ)
        
        def safe_decode(line):
            try:
                return line.decode('utf-8')
            except UnicodeDecodeError:
                return line.decode('latin-1', errors='ignore')
        
        reader = (safe_decode(line) for line in iter(mm.readline, b""))
        chunk = []
        last_log = time.time()

        for line in reader:
            chunk.append(line)
            if len(chunk) >= CHUNK_SIZE:
                results = process_chunk(chunk)
                valid = [res for res in results if res]
                outfile.writelines(valid)
                
                stats['success_lines'] += len(valid)
                stats['error_lines'] += len(chunk) - len(valid)
                stats['bytes_written'] += sum(len(line) for line in valid)
                
                progress.update(len(chunk))
                chunk = []

                # Periodic logging
                if (time.time() - last_log) > LOG_INTERVAL * 60:
                    sys_stats = get_system_stats()
                    print(f"\n[Status @ {format_interval(time.time()-start_time)}] "
                          f"CPU: {sys_stats['cpu_usage']}% | "
                          f"Mem: {sys_stats['mem_used']:.1f}/{sys_stats['mem_total']:.1f}GB | "
                          f"Processed: {stats['success_lines']/1e6:.1f}M lines")
                    last_log = time.time()

        # Process remaining lines
        if chunk:
            results = process_chunk(chunk)
            valid = [res for res in results if res]
            outfile.writelines(valid)
            stats['success_lines'] += len(valid)
            stats['error_lines'] += len(chunk) - len(valid)
            stats['bytes_written'] += sum(len(line) for line in valid)
            progress.update(len(chunk))

        mm.close()
        progress.close()

    # Final report
    total_time = time.time() - start_time
    print(f"\n{' Completion Summary ':=^80}")
    print(f"Total Time:    {format_interval(total_time)}")
    print(f"Lines:         {stats['success_lines']:,} successful | {stats['error_lines']:,} failed")
    print(f"Throughput:    {stats['success_lines']/total_time:,.1f} lines/sec")
    print(f"Output Size:   {stats['bytes_written']/(1024**2):.2f} MB")

if __name__ == '__main__':
    main()