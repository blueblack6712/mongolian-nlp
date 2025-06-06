import json
import numpy as np
import logging
import os
import time
import gc
import signal
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import psutil

# =============================================
# Enhanced Configuration
# =============================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# System resource limits
MAX_MEMORY_PERCENT = 85  # More conservative memory limit
CHUNK_SIZE = 50000       # Adjust according to available memory
FLUSH_INTERVAL = 500     # More frequent flushing to reduce memory
PROCESS_TIMEOUT = 3600   # Process timeout in seconds
BATCH_SIZE = 500         # Smaller batch size for memory safety

# =============================================
# Memory-Aware Data Loader
# =============================================
class MemoryMonitor:
    @staticmethod
    def check_memory():
        vm = psutil.virtual_memory()
        if vm.percent >= MAX_MEMORY_PERCENT:
            logging.critical(f"Memory limit exceeded ({vm.percent}%), terminating safely")
            os.kill(os.getpid(), signal.SIGTERM)

def load_annotated_txt(file_path):
    """Training data loader with progress bar"""
    sentences = []
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading training data") as pbar:
                chunk = []
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    if line.startswith('[sen'):
                        parts = line.split()
                        sentence = []
                        for word_tag in parts[1:]:
                            word, tag = word_tag.rsplit('_', 1)
                            sentence.append((word.strip(), tag.strip()))
                        if sentence:
                            chunk.append(sentence)
                            if len(chunk) >= CHUNK_SIZE:
                                sentences.extend(chunk)
                                del chunk[:]
                                gc.collect()
                                MemoryMonitor.check_memory()
                if chunk:
                    sentences.extend(chunk)
        return sentences
    except Exception as e:
        logging.error(f"Failed to load training data: {str(e)}")
        raise

def stream_jsonl_batches(file_path, batch_size=BATCH_SIZE):
    """Stream JSONL in batches without full memory load"""
    batch = []
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=file_size, unit='B', unit_scale=True, desc="Streaming input"):
                item = json.loads(line)
                if "text" not in item:
                    continue
                tokens = [t.replace('▁', ' ').strip() for t in item["text"].split()]
                batch.append({'original_text': item["text"], 'tokens': tokens})
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                    gc.collect()
                    MemoryMonitor.check_memory()
        if batch:
            yield batch
    except Exception as e:
        logging.error(f"Failed to stream JSONL: {str(e)}")
        raise

# =============================================
# Mongolian POS Tagger (Optimized)
# =============================================
class MongolianPOSTagger:
    def __init__(self, rare_threshold=3, smoothing=1e-3):
        self.tag_list = []
        self.vocab = set()
        self.transition_probs = None
        self.emission_probs = None
        self.rare_threshold = rare_threshold
        self.smoothing = smoothing
        self.punctuation = {'.', ',', '!', '?', ';', ':'}

    def train(self, train_data):
        """Optimized training with memory checks"""
        logging.info("Starting training...")
        start_time = time.time()
        
        # Phase 1: Batch counting
        all_words, all_tags = [], []
        with tqdm(total=len(train_data), desc="Processing training data") as pbar:
            for sentence in train_data:
                tags = [tag for _, tag in sentence]
                words = [word for word, _ in sentence]
                all_tags.extend(tags)
                all_words.extend(words)
                pbar.update(1)
                if pbar.n % 1000 == 0:
                    gc.collect()
                    MemoryMonitor.check_memory()

        # Vectorized operations
        unique_tags, tag_counts = np.unique(all_tags, return_counts=True)
        unique_words, word_counts = np.unique(all_words, return_counts=True)
        
        # Build vocabulary and tag list
        self.vocab = set(unique_words[word_counts > self.rare_threshold])
        self.tag_list = unique_tags[np.argsort(-tag_counts)].tolist()
        
        # Transition matrix
        tag_indices = {tag: idx for idx, tag in enumerate(self.tag_list)}
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        with tqdm(total=len(all_tags)-1, desc="Transition matrix") as pbar:
            for i in range(len(all_tags)-1):
                current = all_tags[i]
                next_tag = all_tags[i+1]
                transition_counts[current][next_tag] += 1
                pbar.update(1)

        self.transition_probs = np.zeros((len(self.tag_list), len(self.tag_list))) + self.smoothing
        for i, tag in enumerate(self.tag_list):
            for next_tag, count in transition_counts[tag].items():
                j = tag_indices[next_tag]
                self.transition_probs[i, j] = count
        row_sums = self.transition_probs.sum(axis=1)[:, np.newaxis]
        self.transition_probs = np.log(self.transition_probs / row_sums)

        # Emission probabilities
        self.emission_probs = {}
        word_tag_counts = defaultdict(lambda: defaultdict(int))
        
        with tqdm(total=len(train_data), desc="Emission matrix") as pbar:
            for sentence in train_data:
                for word, tag in sentence:
                    if word in self.vocab:  # Only track frequent words
                        word_tag_counts[tag][word] += 1
                pbar.update(1)

        for tag in tqdm(self.tag_list, desc="Finalizing emissions"):
            total = sum(word_tag_counts[tag].values()) + self.smoothing * (len(self.vocab) + 1)
            self.emission_probs[tag] = {
                word: np.log((count + self.smoothing) / total)
                for word, count in word_tag_counts[tag].items()
            }

        elapsed = time.time() - start_time
        logging.info(f"Training completed in {elapsed:.2f} seconds")

    def viterbi(self, sentence):
        """Memory-efficient Viterbi"""
        if not sentence:
            return []
            
        T = len(sentence)
        N = len(self.tag_list)
        viterbi = np.full((N, T), -np.inf)
        backpointers = np.zeros((N, T), dtype=int)

        # Initialization
        first_word = sentence[0]
        for i in range(N):
            emission = self.emission_probs[self.tag_list[i]].get(first_word, np.log(self.smoothing))
            viterbi[i, 0] = emission + np.log(1/N)

        # Dynamic programming
        for t in range(1, T):
            current_word = sentence[t]
            for j in range(N):
                trans_probs = self.transition_probs[:, j] + viterbi[:, t-1]
                max_score = np.max(trans_probs)
                best_prev = np.argmax(trans_probs)
                emission = self.emission_probs[self.tag_list[j]].get(current_word, np.log(self.smoothing))
                viterbi[j, t] = max_score + emission
                backpointers[j, t] = best_prev

        # Backtrace
        best_path = [np.argmax(viterbi[:, -1])]
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointers[best_path[0], t])
            
        return [self.tag_list[i] if word not in self.punctuation else "PUN" 
                for i, word in zip(best_path, sentence)]

# =============================================
# Parallel Processing Module (Optimized)
# =============================================
# Global variables for shared parameters
global_tagger = None

def init_worker(tagger_params):
    """Initialize worker with shared parameters"""
    global global_tagger
    global_tagger = MongolianPOSTagger()
    global_tagger.tag_list = tagger_params['tag_list']
    global_tagger.transition_probs = tagger_params['transition_probs']
    global_tagger.emission_probs = tagger_params['emission_probs']
    global_tagger.punctuation = tagger_params['punctuation']

def process_batch(batch):
    """Process batch using shared tagger"""
    try:
        results = []
        for doc in batch:
            if not isinstance(doc, dict) or 'tokens' not in doc:
                continue
            try:
                pos_tags = global_tagger.viterbi(doc['tokens'])
                results.append({
                    "original_text": doc['original_text'],
                    "tokens": doc['tokens'],
                    "pos_tags": pos_tags
                })
            except Exception as e:
                logging.warning(f"Failed processing document: {str(e)}")
                
            if len(results) % 100 == 0:
                gc.collect()
        return results
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        return []

class SafeWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = []
        self.count = 0
        
    def write(self, data):
        self.buffer.append(json.dumps(data, ensure_ascii=False) + '\n')
        self.count += 1
        
        if self.count % FLUSH_INTERVAL == 0:
            self._safe_flush()
            
    def _safe_flush(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.writelines(self.buffer)
                self.buffer = []
                return
            except IOError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logging.warning(f"Write failed (attempt {attempt+1}), waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
                    
    def close(self):
        if self.buffer:
            self._safe_flush()

# =============================================
# Main Program (Optimized)
# =============================================
if __name__ == "__main__":
    # Path configuration
    PATHS = {
        "train": r"dataset.txt"
    }
    
    input_files = [f"/dev/shm/pos_tagged_output_{i}.jsonl" for i in range(11,11)]
    output_files = [f"/dev/shm/pos_tagged_dataset_{i}.jsonl" for i in range(12, 12)]
    
    try:
        # Training Phase
        logging.info("Initializing training...")
        train_data = load_annotated_txt(PATHS["train"])
        tagger = MongolianPOSTagger(rare_threshold=3)
        tagger.train(train_data[:500000])
        
        # Prepare shared parameters
        tagger_params = {
            'tag_list': tagger.tag_list,
            'transition_probs': tagger.transition_probs,
            'emission_probs': tagger.emission_probs,
            'punctuation': tagger.punctuation
        }
        
        # Process each file
        num_cores = max(1, cpu_count() // 2)  # Use half cores for memory
        for idx, (input_file, output_file) in enumerate(zip(input_files, output_files), 1):
            start_time = time.time()
            logging.info(f"Processing file {idx}: {input_file}")
            
            if os.path.exists(output_file):
                os.remove(output_file)
            
            writer = SafeWriter(output_file)
            try:
                with Pool(processes=num_cores, initializer=init_worker, 
                         initargs=(tagger_params,)) as pool:
                    
                    batch_generator = stream_jsonl_batches(input_file)
                    total_batches = os.path.getsize(input_file) // (BATCH_SIZE * 1000)  # Estimate
                    
                    with tqdm(total=total_batches, desc=f"File {idx}") as pbar:
                        for result in pool.imap(process_batch, batch_generator, chunksize=2):
                            for item in result:
                                writer.write(item)
                            pbar.update(1)
                            pbar.set_postfix({
                                "Mem": f"{psutil.virtual_memory().percent}%",
                                "CPU": f"{psutil.cpu_percent()}%"
                            })
                            if pbar.n % 10 == 0:
                                gc.collect()
            finally:
                writer.close()
                gc.collect()
            
            elapsed = time.time() - start_time
            logging.info(f"Finished file {idx} in {elapsed:.2f}s")
        
        logging.info("All files processed successfully!")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise
