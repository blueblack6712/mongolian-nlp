from pathlib import Path
import json

def convert_txt_to_jsonl(input_dir="raw_data", output_dir="converted"):
    """Convert all .txt files to .jsonl format"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    txt_files = list(input_path.glob("*.txt"))
    
    for i, txt_file in enumerate(txt_files):
        jsonl_file = output_path / f"{txt_file.stem}.jsonl"
        
        with open(txt_file, 'r', encoding='utf-8') as f_in, \
             open(jsonl_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if line.strip():
                    json.dump({"content": line.strip()}, f_out, ensure_ascii=False)
                    f_out.write('\n')
        
        print(f"Converted {i+1}/{len(txt_files)}: {txt_file.name} → {jsonl_file.name}")

# Usage:
convert_txt_to_jsonl(input_dir=r"C:\Users\POO\Desktop\mn", output_dir=r"C:\Users\POO\Desktop\mn_converted")