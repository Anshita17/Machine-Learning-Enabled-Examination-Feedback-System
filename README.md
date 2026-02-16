# 📚 Machine Learning-Enabled Examination Feedback System
## Fine-Tuning DistilBERT for Nuanced Academic & Subjective Text Classification

---

## 📌 Project Overview

This research project, published at the **2025 3rd IEEE International Conference (ICIDeA)**, addresses the challenge of analyzing high-volume, unstructured academic data. It presents an end-to-end framework for classifying subjective responses and study materials into 17 distinct categories.

By bridging the gap between classical NLP and modern Transformers, this system:

- 🔎 **Extracts** text from complex PDFs using `PyMuPDF` and `pdfplumber`.
- 🧹 **Normalizes** data through automated cleaning, noise removal (Regex), and linguistic normalization (`spaCy`/`NLTK`).
- 📊 **Benchmarks** performance between traditional ML (TF-IDF) and Transformer architectures.
- 🎯 **Classifies** nuanced and deceptive language patterns using a fine-tuned **DistilBERT** model.
- 🚨 **Analyzes Robustness** by testing model performance against adversarial or evasive textual manipulation.
- 💬 **Generates Feedback** using semantic similarity scoring for actionable academic recommendations.

🔗 **IEEE Publication:** [DOI: 10.1109/ICIDeA64800.2025.10963164](https://doi.org/10.1109/ICIDeA64800.2025.10963164)  
🔗 **Live Model Hub:** [AnshitaPriyadarshini17/distilbert_model_automated_examination_feedback](https://huggingface.co/AnshitaPriyadarshini17/distilbert_model_automated_examination_feedback)

---

## 🎯 Objectives

- Develop a scalable pipeline for classifying complex, unstructured textual data across 17 categories.
- Capture nuanced and "deceptive" language patterns in subjective academic responses.
- Optimize for deployment efficiency (DistilBERT) without sacrificing classification performance.
- Demonstrate a reproducible framework from raw data extraction to feedback generation.

---

## 🏗️ System Architecture



1. **Extraction:** Raw PDF Data handling via `PyMuPDF` and `PyPDF2`.
2. **Preprocessing:** Automated noise removal (`Regex`) and linguistic normalization (`spaCy`).
3. **Embeddings:** Comparing `TF-IDF` baselines against `Transformer` embeddings.
4. **Classification:** Fine-tuned `DistilBERT` via `PyTorch` and `HuggingFace Transformers`.
5. **Evaluation:** Robustness analysis and metric benchmarking (Accuracy, F1, Precision, Recall).
6. **Output:** Semantic similarity scoring and actionable feedback generation.

---

## 🛠️ Technologies Used

### Deep Learning & AI
- **Python 3.9+**
- **PyTorch**
- **Hugging Face Transformers** (DistilBERT)
- **Scikit-Learn** (Traditional ML baselines)

### NLP & Data Engineering
- **spaCy** & **NLTK** (Tokenization & Lemmatization)
- **PyMuPDF** & **pdfplumber** (Text extraction)
- **NumPy** & **Pandas**
- **JSON** (Data structuring)

---

## 🤖 Research Results

- **Efficiency:** DistilBERT is 40% smaller and 60% faster than BERT, while retaining **97% of performance**.

- **Top Category Performance:**  
  - **Governance:** 0.90 F1-Score  
  - **Internal Security:** 0.80 F1-Score  

- **Robustness:** Successfully detected patterns even when vocabulary or structure was altered to simulate adversarial text.

# 💻 How to Use This Project

### 🟢 Load Published Model

The fine-tuned model weights are available via the Hugging Face Hub. This allows you to use the research output directly without local training.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Research-grade model published on Hugging Face
repo_id = "AnshitaPriyadarshini17/distilbert_model_automated_examination_feedback"

model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

