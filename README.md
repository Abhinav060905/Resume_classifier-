# Resume Classification into Job Roles using NLP

A machine learning project that classifies resumes into job role categories using Natural Language Processing techniques.

## Overview

This project takes a resume (PDF or text) and predicts the most suitable job role category:

- Data Scientist
- Software Engineer
- Web Developer
- DevOps Engineer
- UI/UX Designer
- Business Analyst
- ML Engineer

The system uses a TF-IDF based feature extraction pipeline combined with classical ML classifiers.

## Datasets

The project now uses multiple CSV datasets stored in `data/`. During training, `src/train_model.py` loads every available dataset, normalizes the text and label columns into a shared schema, and combines them into one training set.

| File | Rows | Purpose | Columns used in training |
|------|------|---------|--------------------------|
| `resumes_dataset.csv` | 51 | Small curated seed dataset of resumes across the core project roles | `resume_text`, `job_role` |
| `Resume.csv` | 2,484 | Larger resume corpus with category labels | `Resume_str`, `Category` |
| `dataset.csv` | 10,174 | Resume-role dataset with extra decision metadata | `Resume`, `Role` |
| `validation.csv` | 37,740 | Resume/job matching dataset used as an additional weakly supervised source | `text`, `label`, `resume_domain` |

### Original seed dataset distribution

`resumes_dataset.csv` contains the original hand-curated examples for the 7 target roles:

| Job Role | Count |
|----------|-------|
| Data Scientist | 9 |
| Software Engineer | 7 |
| Web Developer | 7 |
| DevOps Engineer | 7 |
| UI/UX Designer | 7 |
| Business Analyst | 7 |
| ML Engineer | 7 |

### Dataset notes

- `Resume.csv` categories are mapped into unified job-role labels before training.
- `dataset.csv` role names are normalized because the source file uses inconsistent casing and naming.
- `validation.csv` stores resume/job pairs in one `text` field separated by `[SEP]`; the training script keeps only positive matches (`label = 1`) and extracts the resume side.
- Rows that cannot be mapped into the shared role taxonomy are dropped during preprocessing.

## ML Pipeline

1. **Data Loading** - Load and merge all available CSV datasets from `data/`
2. **Text Preprocessing** - Lowercase, remove punctuation/URLs/emails, stopword removal, lemmatization
3. **Label Normalization** - Map source-specific labels into a unified job-role taxonomy
4. **Feature Extraction** - TF-IDF vectorization with unigrams and bigrams
5. **Model Training** - Train and compare 4 classifiers:
   - Logistic Regression
   - Naive Bayes (Multinomial)
   - Random Forest
   - Linear SVM
6. **Evaluation** - Accuracy, Precision, Recall, F1 Score, Cross-Validation, Confusion Matrix
7. **Model Saving** - Best model and vectorizer saved as pickle files
8. **Prediction** - Classify new resumes via CLI or web interface

## Project Structure

```text
resume-classifier/
|-- data/
|   |-- resumes_dataset.csv
|   |-- Resume.csv
|   |-- dataset.csv
|   `-- validation.csv
|-- src/
|   |-- preprocess.py
|   |-- feature_extraction.py
|   |-- train_model.py
|   `-- predict.py
|-- app/
|   `-- app.py
|-- models/
|   |-- classifier.pkl
|   `-- vectorizer.pkl
|-- utils/
|   `-- pdf_parser.py
|-- plots/
|   |-- model_comparison.png
|   |-- confusion_matrix.png
|   `-- class_distribution.png
|-- README.md
`-- requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd src
python train_model.py
```

This will:
- Load and combine all supported datasets from `data/`
- Normalize source labels into the shared job-role set
- Preprocess the resume text
- Train 4 different models
- Print evaluation metrics for each
- Save the best model to `models/classifier.pkl`
- Generate comparison plots in `plots/`

### 3. Predict from command line

```bash
# using a PDF resume
cd src
python predict.py path/to/resume.pdf

# or run with demo text (no arguments)
python predict.py
```

### 4. Run the web interface

```bash
cd resume-classifier
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.

## Why TF-IDF?

TF-IDF (Term Frequency - Inverse Document Frequency) works well for resume classification because:

- It identifies words that are important in a specific resume but not common across all resumes
- Domain-specific terms like "tensorflow", "kubernetes", "figma" become strong signals for their respective roles
- It handles the varying lengths of resumes naturally
- Combined with bigrams, it can capture phrases like "machine learning" or "user experience"

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies
