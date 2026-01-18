## **Mongolian NLP Pipeline for POS Tagging**

## 📋 Project Overview
This project implements a comprehensive Natural Language Processing pipeline for Modern Mongolian (Cyrillic script), focusing on Part-of-Speech tagging, tokenization, and model evaluation. The system combines statistical methods (Bigram HMM-Viterbi) with modern transformer fine-tuning (BERT and LLaMA) to address the unique challenges of Mongolian's agglutinative morphology.

```markdown
## 📁 Project Structure
mongolian-nlp-pipeline/
├── src/                    # Main source code directory
│   ├── config.json        # Configuration file for model parameters
│   ├── evaluation.py      # Evaluation metrics and performance analysis
│   ├── graph.py           # Visualization and graph generation
│   ├── requirements       # Python dependencies list
│   ├── results.py         # Results processing and formatting
│   ├── 1_Models_token.py  # Model tokenization implementation
│   ├── 2_txt2jsonl.py     # Convert text files to JSONL format
│   ├── 3_merging_jsonl.py # Merge multiple JSONL files
│   ├── 4_data_cleaning.py # Data cleaning and preprocessing
│   ├── 5.1_morph.py       # Morphological analysis implementation
│   ├── 5.1_morph+hf+nltk+spm.py # Combined tokenization methods
│   ├── 5.2_spm-jsonl.py   # SentencePiece tokenization for JSONL
│   ├── 5.3_spm-jsonl.py   # Additional SentencePiece processing
│   ├── 5.3_split4.py      # Data splitting utilities
│   ├── 5.4_split3.py      # Alternative data splitting
│   ├── 5.5_merge78.py     # Merge processed datasets
│   ├── 5_bert_tokenize.py # BERT tokenization implementation
│   ├── 6_postag.py        # Part-of-Speech tagging core logic
│   ├── 7_annotated.py     # Annotation processing utilities
│   ├── 8_data_split.py    # Dataset splitting for train/val/test
│   └── 10_ft_llama.py     # LLaMA fine-tuning implementation
├── notebooks/             # Jupyter notebooks for interactive development
│   ├── main.ipynb        # Main pipeline execution notebook
│   ├── bert_ft.ipynb     # BERT fine-tuning notebook
│   └── llama_ft.ipynb    # LLaMA fine-tuning notebook
├── data/                  # Data storage directory
│   ├── datasets/         # Raw and processed datasets
│   └── results/          # Model outputs and evaluation results
└── README.md             # This file
```


## 🎯 Key Features
- **Mongolian-specific preprocessing**: Handles Cyrillic script, vowel harmony, and agglutinative structures
- **Hybrid POS tagging**: Combines statistical HMM models with neural transformer fine-tuning
- **Multiple tokenization strategies**: Evaluates and compares SentencePiece, regex/NLTK, and morphological approaches
- **Efficient processing**: Parallel batch processing and memory-optimized data loading
- **Comparative evaluation**: Benchmarks against existing Mongolian NLP methods

## 🏗️ Pipeline Architecture

### Phase 1: Data Processing (`src/2_` to `src/5.5_`)
1. **Text to JSONL Conversion** (`2_txt2jsonl.py`): Convert raw text files to JSONL format
2. **Dataset Merging** (`3_merging_jsonl.py`): Combine multiple JSONL datasets
3. **Data Cleaning** (`4_data_cleaning.py`): Clean and normalize Mongolian text
4. **Tokenization Methods** (`5.1_` to `5.5_`): Implement and compare different tokenization strategies
   - Morphological analysis (`5.1_morph.py`)
   - Combined tokenizers (`5.1_morph+hf+nltk+spm.py`)
   - SentencePiece processing (`5.2_spm-jsonl.py`, `5.3_spm-jsonl.py`)
   - Data splitting utilities (`5.3_split4.py`, `5.4_split3.py`)
   - Dataset merging (`5.5_merge78.py`)

### Phase 2: Model Implementation (`src/6_` to `src/10_`)
1. **BERT Tokenization** (`5_bert_tokenize.py`): BERT-specific tokenization
2. **POS Tagging Core** (`6_postag.py`): Bigram HMM-Viterbi implementation
3. **Annotation Processing** (`7_annotated.py`): Handle annotated datasets
4. **Dataset Splitting** (`8_data_split.py`): Create train/validation/test splits
5. **LLaMA Fine-tuning** (`10_ft_llama.py`): LLaMA model adaptation

### Phase 3: Evaluation & Results (`src/evaluation.py`, `src/results.py`)
1. **Performance Evaluation** (`evaluation.py`): Calculate metrics and benchmarks
2. **Results Processing** (`results.py`): Format and analyze model outputs
3. **Visualization** (`graph.py`): Generate performance graphs and charts

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/mongolian-nlp-pipeline.git
cd mongolian-nlp-pipeline

# Install dependencies
pip install -r src/requirements
````

### Running the Pipeline

#### Option 1: Using Notebooks (Interactive)

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook notebooks/
   ```
2. Open `main.ipynb` for the complete pipeline
3. For fine-tuning specific models:

   * Use `bert_ft.ipynb` for BERT fine-tuning
   * Use `llama_ft.ipynb` for LLaMA fine-tuning

#### Option 2: Using Python Scripts (Batch Processing)

1. **Data Preparation**:

   ```bash
   # Convert text to JSONL
   python src/2_txt2jsonl.py --input data/datasets/raw --output data/datasets/processed

   # Clean the data
   python src/4_data_cleaning.py --input data/datasets/processed --output data/datasets/cleaned
   ```

2. **Tokenization**:

   ```bash
   # Run morphological tokenization
   python src/5.1_morph.py --input data/datasets/cleaned --output data/datasets/tokenized

   # Or use combined tokenizers
   python src/5.1_morph+hf+nltk+spm.py --input data/datasets/cleaned
   ```

3. **POS Tagging**:

   ```bash
   # Train HMM POS tagger
   python src/6_postag.py --train data/datasets/tokenized/train.jsonl --model data/models/hmm_tagger.pkl

   # Run inference
   python src/6_postag.py --model data/models/hmm_tagger.pkl --input data/datasets/tokenized/test.jsonl --output data/results/tagged_test.jsonl
   ```

4. **Model Fine-tuning**:

   ```bash
   # Fine-tune LLaMA
   python src/10_ft_llama.py --train data/datasets/tokenized/train.jsonl --val data/datasets/tokenized/val.jsonl --output_dir data/models/llama_finetuned
   ```

### Configuration

Edit `src/config.json` to modify:

* Model hyperparameters
* Tokenization settings
* Training configurations
* Path configurations

## 📊 Performance Results

### Model Performance

| Model              | Accuracy | Training Loss | Validation Loss | F1 Score |
| ------------------ | -------- | ------------- | --------------- | -------- |
| BERT               | 97%      | 0.0744        | 0.0754          | 0.96     |

### Tokenization Comparison

* **SentencePiece**: Best performance with linguistically meaningful subword units
* **Regex/NLTK**: Unreliable for Mongolian suffix handling
* **Morphological Splitting**: Linguistically motivated but impractical for POS tagging

## 🧪 Experimental Setup

### Datasets Used

1. **OSCAR-corpus**: 6.9GB web-scraped data
2. **MN News**: 6.6GB Mongolian news dataset
3. **MN-text**: 3.2GB statistical and neural MT data

### Evaluation Metrics

* **Token-level accuracy**: Percentage of correctly tagged tokens
* **Macro F1**: Harmonic mean of precision and recall
* **Word Error Rate (WER)**: For ASR components (22% reduction with transfer learning)

## 🎯 Challenges Addressed

### Linguistic Challenges

* **Agglutinative morphology**: Suffix chains and high morpheme density
* **Vowel harmony**: Context-dependent vowel changes
* **Case system**: 8 grammatical cases requiring explicit modeling
* **Script variations**: Cyrillic vs. Traditional Mongolian script

### Technical Challenges

* **Data scarcity**: Limited annotated corpora (110K → 260K words expanded)
* **Computational constraints**: CUDA out-of-memory on RTX 4090
* **Tokenization complexity**: Suffix ambiguity and OOV handling

## 🔄 Dataset Adaptation

### Modified Dataset

We adapted and enhanced the Mongolian POS tagging dataset from Ganchimeg's repository:

* **Original Source**: [ganchimegl/POS-tagger-for-Mongolian](https://github.com/ganchimegl/POS-tagger-for-Mongolian)
* **Original Size**: ~100,000 word tokens (5,000 sentences)
* **Our Enhancements**:

  * Expanded dataset to 260,000 words to cover diverse morphological patterns
  * Fixed inconsistent punctuation tagging (converted "VS" to "PUN")
  * Added UTF-8 encoding support for Cyrillic characters
  * Implemented memory-optimized loading for large-scale processing
  * Extended with additional Mongolian text corpora for robustness

## 🙏 Acknowledgments

### Code and Dataset Credits

1. **Ganchimeg's HMM Tagger**: We thank Ganchimeg for their open-source Mongolian POS tagger implementation. Our work builds upon and extends their HMM approach with significant improvements in scalability and accuracy.

   * Repository: [POS-tagger-for-Mongolian](https://github.com/ganchimegl/POS-tagger-for-Mongolian)
   * Original work provided foundational HMM implementation and initial dataset

2. **Data Providers**:

   * Inner Mongolia University for MnTTS corpus
   * Hugging Face for OSCAR and model repositories
   * Yandex for MN News dataset
   * statmt.org for MN-text dataset

3. **Model Resources**:

   * Google for BERT-base-multilingual-cased
   * Meta for LLaMA-2-3B
   * Hugging Face for transformers library and model hosting

### Computational Resources

* Autodl for providing GPU instances (RTX 4090 with 24GB VRAM)
* Team members' local machines for development and testing


## 📚 References

### Academic Papers

1. Liu et al. (2020) - Inner-word and Out-word Features for Mongolian Morphological Segmentation
2. Jaimai & Chimeddorj (2009) - Part of Speech Tagging for Mongolian Corpus
3. Munkhjargal et al. (2015) - Named Entity Recognition for Mongolian Language

## 👥 Contributing

We welcome contributions in:

* Data annotation and corpus expansion
* Model optimization and efficiency improvements
* Extension to additional Mongolian dialects
* Documentation and tutorial creation


*This project addresses critical gaps in Mongolian NLP, providing practical tools for researchers and developers working with this unique and challenging language. We acknowledge the foundational work of the Mongolian NLP community and hope our contributions further advance this field.*
