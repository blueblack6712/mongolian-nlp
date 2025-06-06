import json
import os

# Configuration
INPUT_JSONL = r"C:\Users\POO\merged_mongolian_dataset.jsonl"  # Your merged JSONL file
OUTPUT_TXT = r"C:\Users\POO\merged_mongolian_dataset.txt"  # Output TXT file

def jsonl_to_bert_txt():
    with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
         open(OUTPUT_TXT, "w", encoding="utf-8") as outfile:
        
        for line_number, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract Mongolian text - handles different JSON structures
                text = data.get("text") or data.get("content") or data.get("body")
                
                if text:
                    # Clean text and write to TXT
                    cleaned_text = text.replace("\n", " ").strip()  # Remove newlines within documents
                    outfile.write(cleaned_text + "\n")  # One document per line
                else:
                    print(f"Warning: No text found in line {line_number}")
                    
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {line_number}")
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {str(e)}")

if __name__ == "__main__":
    jsonl_to_bert_txt()
    print(f"Conversion complete. TXT file saved to: {OUTPUT_TXT}")