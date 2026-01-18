import json
import os
import math

def split_jsonl_safely():
    """
    专用分割函数，硬编码文件路径
    输入：/dev/shm/spm_mongolian_dataset.jsonl
    输出：/dev/shm/pos_tagged_output_{1-4}.jsonl
    """
    input_path = "/dev/shm/spm_mongolian_dataset.jsonl"
    output_prefix = "/dev/shm/pos_tagged_output"
    num_parts = 4
    
    # 验证输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")

    # 第一阶段：统计总行数
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # 计算每个文件的行数
    lines_per_file = total_lines // num_parts
    remainder = total_lines % num_parts
    
    # 第二阶段：执行分割
    part_num = 1
    current_line = 0
    output_file = None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 创建新文件的条件
            if output_file is None or (
                current_line >= lines_per_file + (1 if part_num <= remainder else 0)
            ):
                if output_file:
                    output_file.close()
                output_path = f"{output_prefix}_{part_num}.jsonl"
                output_file = open(output_path, 'w', encoding='utf-8')
                part_num += 1
                current_line = 0
                lines_in_this_file = lines_per_file + (1 if part_num <= remainder else 0)
            
            # 写入并验证JSON格式
            try:
                json.loads(line)  # 格式验证
                output_file.write(line)
                current_line += 1
            except json.JSONDecodeError:
                print(f"发现损坏行，已跳过：{line.strip()}")
    
    if output_file:
        output_file.close()
    print("分割完成！生成文件：")
    print("\n".join([f"/dev/shm/pos_tagged_output_{i}.jsonl" for i in range(1, 5)]))

if __name__ == "__main__":
    split_jsonl_safely()