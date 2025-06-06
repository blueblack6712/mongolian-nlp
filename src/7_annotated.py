import ujson
import gc
import os
import psutil
import signal

# ⚙️ 配置参数
DATASET_PATH = "/dev/shm/pos_tagged_dataset_1.jsonl"
BERT_VOCAB = "/dev/shm/bert-base-multiligual-cased/vocab.txt"
OUTPUT_BERT = "bert_annotations.jsonl"
OUTPUT_LLAMA = "llama_annotations.jsonl"
PAGE_SIZE = 200  # 初始分页大小
MEMORY_THRESHOLD = 80  # 内存警戒百分比

# 🔥 内存守护系统
class MemoryGuardian:
    @staticmethod
    def get_memory_usage():
        """获取当前内存使用百分比"""
        return psutil.virtual_memory().percent
    
    @staticmethod
    def memory_safety_check():
        """内存安全检查"""
        if MemoryGuardian.get_memory_usage() > MEMORY_THRESHOLD:
            print(f"🚨 内存告警 ({MemoryGuardian.get_memory_usage()}%)，启动紧急清理...")
            gc.collect()
            return True
        return False

# 🔥 崩溃防护型BERT处理器
class CrashSafeBertProcessor:
    def __init__(self, vocab_path):
        """安全初始化"""
        self.root = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                if (word := line.strip()):
                    node = self.root
                    for char in word:
                        node = node.setdefault(char, {})
                    node["__end__"] = None
    
    def _adaptive_page_size(self):
        """动态调整分页大小"""
        global PAGE_SIZE
        mem_usage = MemoryGuardian.get_memory_usage()
        if mem_usage > 75:
            PAGE_SIZE = max(50, PAGE_SIZE // 2)
        elif mem_usage < 40:
            PAGE_SIZE = min(1000, PAGE_SIZE * 2)
        return PAGE_SIZE

    def tokenize(self, word):
        """安全分词流"""
        start = 0
        while start < len(word):
            max_len = self._find_longest(word, start)
            if max_len == 0:
                yield "[UNK]"
                start += 1
            else:
                token = word[start:start+max_len]
                if start > 0:
                    token = "##" + token
                yield token
                start += max_len

    def process(self, input_path, output_path):
        """崩溃防护处理流程"""
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            
            page = []
            for line_num, line in enumerate(fin, 1):
                try:
                    # 内存紧急检查
                    if MemoryGuardian.memory_safety_check():
                        # 立即写入当前页
                        if page:
                            fout.write("\n".join(page) + "\n")
                            page = []
                            gc.collect()
                    
                    # 动态分页调整
                    current_page_size = self._adaptive_page_size()
                    
                    # 数据处理
                    data = ujson.loads(line)
                    tokens = data["tokens"]
                    tags = data["pos_tags"]
                    
                    # 流式生成输出
                    bert_tokens = []
                    aligned_tags = []
                    for w, t in zip(tokens, tags):
                        subs = list(self.tokenize(w))
                        bert_tokens.extend(subs)
                        aligned_tags.append(t)
                        aligned_tags.extend(["X"]*(len(subs)-1))
                    
                    page.append(ujson.dumps({
                        "tokens": bert_tokens,
                        "pos_tags": aligned_tags
                    }, ensure_ascii=False))
                    
                    # 强制内存清理
                    del data, tokens, tags, bert_tokens, aligned_tags
                    if len(page) >= current_page_size:
                        fout.write("\n".join(page) + "\n")
                        page = []
                        print(f"BERT: {line_num} lines | Page: {current_page_size} | Mem: {MemoryGuardian.get_memory_usage()}%")
                        gc.collect()
                
                except Exception as e:
                    print(f"🚑 行 {line_num} 错误跳过: {str(e)}")
                    del line  # 确保错误行内存释放
                    gc.collect()
                    continue
            
            if page:
                fout.write("\n".join(page) + "\n")

# ⚡ 安全型LLaMA处理器
class CrashSafeLlamaProcessor:
    def process(self, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            
            page = []
            for line_num, line in enumerate(fin, 1):
                # 内存警戒检查
                if MemoryGuardian.memory_safety_check() and page:
                    fout.write("\n".join(page) + "\n")
                    page = []
                    gc.collect()
                
                data = ujson.loads(line)
                annotated = " ".join(f"{w}[{t}]" for w,t in zip(data["tokens"], data["pos_tags"]))
                
                page.append(ujson.dumps({
                    "prompt": "Annotate:",
                    "completion": f" {annotated}"
                }, ensure_ascii=False))
                
                if len(page) >= self._adaptive_page_size():
                    fout.write("\n".join(page) + "\n")
                    page = []
                    print(f"LLaMA: {line_num} lines | Mem: {MemoryGuardian.get_memory_usage()}%")
                    gc.collect()
                
                del data, annotated
                gc.collect()
            
            if page:
                fout.write("\n".join(page) + "\n")

def system_cleanup():
    """最后一道内存防线"""
    print("🚒 启动系统级清理...")
    os.system('sync && echo 3 > /proc/sys/vm/drop_caches')  # Linux系统缓存清理

if __name__ == "__main__":
    # 注册崩溃保护
    signal.signal(signal.SIGINT, lambda *_: system_cleanup())
    
    try:
        print("🛡️ BERT处理启动 (崩溃防护模式)")
        bert_processor = CrashSafeBertProcessor(BERT_VOCAB)
        bert_processor.process(DATASET_PATH, OUTPUT_BERT)
        del bert_processor
        gc.collect()
        
        print("\n🛡️ LLaMA处理启动 (崩溃防护模式)")
        llama_processor = CrashSafeLlamaProcessor()
        llama_processor.process(DATASET_PATH, OUTPUT_LLAMA)
        
    finally:
        system_cleanup()
        print("✅ 全流程完成！")