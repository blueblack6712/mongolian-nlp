import json
import os
import glob

# Use raw string and glob to handle paths safely
input_folder = r"C:\Users\POO\Desktop\test"
jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

output_file = r"test_merged_mongolian_dataset.jsonl"

TEXT_KEYS = ["content"]

with open(output_file, "w", encoding="utf-8") as f_out:
    for file_path in jsonl_files:
        try:
            # Use UTF-8-sig encoding to handle BOM if present
            with open(file_path, "r", encoding="utf-8-sig") as f_in:
                for line_number, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        for key in TEXT_KEYS:
                            if key in data:
                                text = data[key].strip()
                                if text:
                                    f_out.write(text + "\n")
                                break
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {os.path.basename(file_path)} line {line_number}")
                    except KeyError:
                        print(f"No valid text key found in {os.path.basename(file_path)} line {line_number}")
                        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except UnicodeDecodeError:
            print(f"Encoding error in file: {file_path}")

print("Merging completed successfully!")