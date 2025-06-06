# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-base-mongolian-cased")
model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-base-mongolian-cased")

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-base-mongolian-cased")
model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-base-mongolian-cased")