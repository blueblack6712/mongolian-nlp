import json
import os
import glob

#input_folder = r"C:\Users\POO\Desktop\mn_converted"
input_folder = r"C:\Users\POO\Desktop\test"
jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
#output_file = r"merged_mongolian_dataset.jsonl"
output_file = r"merged_test.jsonl"


def is_valid_json(line):
    """Validate if a line contains valid JSON"""
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False

with open(output_file, "w", encoding="utf-8") as f_out:
    line_counter = 0
    for file_path in jsonl_files:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f_in:
                print(f"Merging {os.path.basename(file_path)}...")
                
                for line_number, line in enumerate(f_in, 1):
                    # Keep original line endings and whitespace
                    original_line = line.rstrip('\n')  # Remove only trailing newline
                    
                    if original_line.strip() == "":  # Skip empty lines
                        continue
                        
                    if is_valid_json(original_line):
                        f_out.write(original_line + "\n")  # Add standardized newline
                        line_counter += 1
                    else:
                        print(f"Skipped invalid JSON at {os.path.basename(file_path)} line {line_number}")
                        
                print(f"Added {line_number} lines from {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

print(f"\nMerging complete! Final line count: {line_counter}")
print(f"Output saved to: {output_file}")
