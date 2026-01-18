import sentencepiece as spm
import time
import json
from tqdm import tqdm  # For progress bar
from multiprocessing import Pool, cpu_count
import os

# Load the SentencePiece model
s = spm.SentencePieceProcessor()
s.Load('mn_cased.model')

# Input and output file paths (use raw strings or double backslashes for Windows)
input_file = r'C:\Users\POO\cleaned_mongolian_dataset.jsonl'  # Your 15GB .jsonl dataset
output_file = r'C:\Users\POO\spm_mongolian_test.jsonl'  # Output tokenized dataset

# Function to tokenize a single line (JSON object)
def tokenize_line(line):
    try:
        # Parse the JSON object
        data = json.loads(line.strip())
        text = data.get('text', '')  # Extract the 'text' field
        # Tokenize the text
        tokens = s.EncodeAsPieces(text)
        # Return the tokenized text as a JSON object
        return json.dumps({'text': ' '.join(tokens)}, ensure_ascii=False) + '\n'
    except Exception as e:
        print(f"Error tokenizing line: {line.strip()}. Error: {e}")
        return None

# Function to process a chunk of lines
def process_chunk(chunk):
    return [tokenize_line(line) for line in chunk]

# Main function
def main():
    start_time = time.time()

    # Determine the number of CPU cores
    num_cores = cpu_count()

    # Read the input file and process in chunks
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the file in chunks
        chunk_size = 10000  # Adjust based on memory availability
        chunk = []
        for line in tqdm(infile, desc="Tokenizing"):
            chunk.append(line)
            if len(chunk) >= chunk_size:
                # Process the chunk in parallel
                with Pool(num_cores) as pool:
                    results = pool.map(tokenize_line, chunk)
                # Write results to the output file
                for result in results:
                    if result:
                        outfile.write(result)
                chunk = []  # Reset the chunk

        # Process the remaining lines in the last chunk
        if chunk:
            with Pool(num_cores) as pool:
                results = pool.map(tokenize_line, chunk)
            for result in results:
                if result:
                    outfile.write(result)

    end_time = time.time()
    print(f"Finished tokenizing in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()