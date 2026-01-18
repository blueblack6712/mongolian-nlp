import json
import os

def split_large_jsonl_by_bytes(input_path, output_prefix, num_parts=3, max_target_gb=3):
    """
    分割指定文件为多个小文件，每个文件按完整行写入，不在行中途切分，
    并尽可能使每个文件字节数平均，目标为总字节数/num_parts，但不超过 max_target_gb (单位GB)。
    输出文件名为: {output_prefix}_5.jsonl, {output_prefix}_6.jsonl, {output_prefix}_7.jsonl
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")
    
    total_size = os.path.getsize(input_path)
    # 计算初步目标大小：总大小除以输出文件数
    base_target = total_size // num_parts
    # 限制单个文件大小不超过 max_target_gb
    max_target_bytes = max_target_gb * (1024**3)
    target_size = base_target if base_target <= max_target_bytes else max_target_bytes
    
    print(f"总文件大小: {total_size/(1024**3):.2f}GB, 每个输出目标大小: {target_size/(1024**3):.2f}GB")
    
    part_num = 5  # 输出文件名从 _5 开始
    current_bytes = 0
    output_file = None
    output_files = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_bytes = len(line.encode('utf-8'))
            # 如果当前没有打开文件，则打开新的
            if output_file is None:
                output_path = f"{output_prefix}_{part_num}.jsonl"
                output_files.append(output_path)
                output_file = open(output_path, 'w', encoding='utf-8')
                current_bytes = 0
                part_num += 1
            
            # 如果当前文件已有内容，并且添加本行后会超过目标大小，则关闭当前文件并开启新文件
            # 注意：如果单行本身超过 target_size，则仍然写入当前文件
            if current_bytes > 0 and (current_bytes + line_bytes) > target_size:
                output_file.close()
                output_file = None
                # 开启新文件
                output_path = f"{output_prefix}_{part_num}.jsonl"
                output_files.append(output_path)
                output_file = open(output_path, 'w', encoding='utf-8')
                current_bytes = 0
                part_num += 1
            
            # 验证JSON格式并写入完整行
            try:
                json.loads(line)  # 格式验证
                output_file.write(line)
                current_bytes += line_bytes
            except json.JSONDecodeError:
                print(f"发现损坏行，已跳过：{line.strip()}")
    
    if output_file:
        output_file.close()
    
    return output_files

def main():
    input_path = "/dev/shm/pos_tagged_dataset_2.jsonl"
    output_prefix = "/dev/shm/pos_tagged_dataset"
    
    # 检查输入文件大小，如果小于3GB则不分割
    file_size_gb = os.path.getsize(input_path) / (1024**3)
    if file_size_gb <= 1:
        print(f"文件 {input_path} 已经小于3GB ({file_size_gb:.2f}GB)，无需分割")
        return
    
    print(f"开始分割文件: {input_path} ({file_size_gb:.2f}GB)")
    output_files = split_large_jsonl_by_bytes(input_path, output_prefix, num_parts=3, max_target_gb=3)
    
    print("\n分割完成！生成文件：")
    for file in output_files:
        size_gb = os.path.getsize(file) / (1024**3)
        print(f"{file} ({size_gb:.2f}GB)")

if __name__ == "__main__":
    main()
