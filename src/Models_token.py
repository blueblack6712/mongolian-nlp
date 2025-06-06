from transformers import AutoTokenizer

# Example Mongolian sentence
text = "Монгол хэлний өгүүлбэр."

# Tokenize with Llama2-3B
llama_tokenizer = AutoTokenizer.from_pretrained("winglian/Llama-2-3b-hf")
llama_tokens = llama_tokenizer.tokenize(text)
print("Llama2-3B Tokens:", llama_tokens)

deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
deepseek_tokens = deepseek_tokenizer.tokenize(text)
print("Deepseek-V3 Tokens:", deepseek_tokens)

# Tokenize with BERT (multilingual)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_tokens = bert_tokenizer.tokenize(text)
print("BERT-base-multilingual-cased Tokens:", bert_tokens)