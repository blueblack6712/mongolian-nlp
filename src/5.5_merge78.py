import os

def merge_jsonl_files(source_file, target_file):
    """
    将 source_file 的内容逐行追加到 target_file 的末尾，确保每个 JSON 对象占一行，
    不会和原有内容合并在同一行。
    """
    # 检查文件是否存在
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件 {source_file} 不存在")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"目标文件 {target_file} 不存在")
    
    # 检查目标文件最后是否以换行结束
    with open(target_file, 'rb+') as tf:
        tf.seek(0, os.SEEK_END)
        if tf.tell() > 0:
            tf.seek(-1, os.SEEK_END)
            last_char = tf.read(1)
            if last_char != b'\n':
                tf.write(b'\n')
    
    # 追加源文件内容到目标文件
    with open(target_file, 'a', encoding='utf-8') as tf, \
         open(source_file, 'r', encoding='utf-8') as sf:
        for line in sf:
            # 确保每一行以换行符结束
            if not line.endswith('\n'):
                line += '\n'
            tf.write(line)

    print(f"已将 {source_file} 的内容成功合并到 {target_file} 中。")

def main():
    source_file = "/dev/shm/pos_tagged_dataset_new.jsonl"
    target_file = "/dev/shm/pos_tagged_dataset_7.jsonl"
    
    merge_jsonl_files(source_file, target_file)
    
if __name__ == "__main__":
    main()
