import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # so it doesn't try to open a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import preprocess_dataset
from feature_extraction import create_tfidf_features, save_vectorizer


# mapping categories from all datasets into unified job roles
# some datasets use different names for similar roles so we group them together

RESUME_CSV_MAPPING = {
    'INFORMATION-TECHNOLOGY': 'IT',
    'ENGINEERING': 'Engineering',
    'BUSINESS-DEVELOPMENT': 'Business Analyst',
    'CONSULTANT': 'Business Analyst',
    'HR': 'HR',
    'DESIGNER': 'UI/UX Designer',
    'DIGITAL-MEDIA': 'Digital Media',
    'FINANCE': 'Finance',
    'ACCOUNTANT': 'Finance',
    'BANKING': 'Finance',
    'HEALTHCARE': 'Healthcare',
    'SALES': 'Sales & Marketing',
    'PUBLIC-RELATIONS': 'Sales & Marketing',
    'TEACHER': 'Education',
    'ARTS': 'Arts',
    'CHEF': 'Chef',
    'FITNESS': 'Fitness',
    'AVIATION': 'Aviation',
    'CONSTRUCTION': 'Construction',
    'ADVOCATE': 'Legal',
    'APPAREL': 'Apparel',
    'AGRICULTURE': 'Agriculture',
    'AUTOMOBILE': 'Automobile',
    'BPO': 'BPO',
}

DATASET_CSV_MAPPING = {
    # normalize the casing first, then map
    'data scientist': 'Data Scientist',
    'software engineer': 'Software Engineer',
    'software developer': 'Software Engineer',
    'product manager': 'Product Manager',
    'data engineer': 'Data Engineer',
    'data analyst': 'Data Analyst',
    'ui engineer': 'UI/UX Designer',
    'ui designer': 'UI/UX Designer',
    'ui/ux designer': 'UI/UX Designer',
    'ux designer': 'UI/UX Designer',
    'devops engineer': 'DevOps Engineer',
    'machine learning engineer': 'ML Engineer',
    'business analyst': 'Business Analyst',
    'full stack developer': 'Web Developer',
    'cloud architect': 'Cloud & Infrastructure',
    'cloud engineer': 'Cloud & Infrastructure',
    'cybersecurity analyst': 'Cybersecurity',
    'cybersecurity specialist': 'Cybersecurity',
    'database administrator': 'Database Administrator',
    'system administrator': 'System Administrator',
    'network engineer': 'Network Engineer',
    'game developer': 'Game Developer',
    'mobile app developer': 'Mobile Developer',
    'ar/vr developer': 'AR/VR Developer',
    'blockchain developer': 'Blockchain Developer',
    'qa engineer': 'QA Engineer',
    'ai researcher': 'AI/ML Research',
    'ai engineer': 'AI/ML Research',
    'content writer': 'Content Writer',
    'digital marketing specialist': 'Digital Media',
    'e-commerce specialist': 'Sales & Marketing',
    'human resources specialist': 'HR',
    'hr specialist': 'HR',
    'it support specialist': 'IT Support',
    'data architect': 'Data Engineer',
    'robotics engineer': 'Robotics Engineer',
    'graphic designer': 'UI/UX Designer',
    'project manager': 'Product Manager',
}

VALIDATION_CSV_MAPPING = {
    # 'technology' is too broad - it includes android devs, web devs, etc.
    # dropping it to avoid flooding the Software Engineer class
    # 'technology': 'Software Engineer',
    'healthcare': 'Healthcare',
    'finance': 'Finance',
    'sales_marketing': 'Sales & Marketing',
    'design': 'UI/UX Designer',
    'legal': 'Legal',
    'manufacturing_operations': 'Manufacturing',
    'education': 'Education',
    'retail_hospitality': 'Retail & Hospitality',
    'human_resources': 'HR',
    'government_nonprofit': 'Government',
    'construction_real_estate': 'Construction',
    'media_entertainment': 'Digital Media',
}


def load_original_dataset(data_dir):
    """load our original synthetic dataset"""
    path = os.path.join(data_dir, 'resumes_dataset.csv')
    if not os.path.exists(path):
        print("Original dataset not found, skipping...")
        return pd.DataFrame(columns=['resume_text', 'job_role'])

    df = pd.read_csv(path)
    # already has the right column names
    print(f"[resumes_dataset.csv] Loaded {len(df)} resumes")
    return df[['resume_text', 'job_role']]


def load_resume_csv(data_dir):
    """
    load Resume.csv - has columns ID, Resume_str, Resume_html, Category
    we only need Resume_str and Category
    """
    path = os.path.join(data_dir, 'Resume.csv')
    if not os.path.exists(path):
        print("Resume.csv not found, skipping...")
        return pd.DataFrame(columns=['resume_text', 'job_role'])

    df = pd.read_csv(path)
    df = df[['Resume_str', 'Category']].copy()
    df.columns = ['resume_text', 'job_role']

    # map categories to our unified roles
    df['job_role'] = df['job_role'].map(RESUME_CSV_MAPPING)
    df = df.dropna(subset=['job_role', 'resume_text'])

    print(f"[Resume.csv] Loaded {len(df)} resumes")
    return df


def load_dataset_csv(data_dir):
    """
    load dataset.csv - has Role and Resume columns
    some roles have inconsistent casing so we normalize first
    """
    path = os.path.join(data_dir, 'dataset.csv')
    if not os.path.exists(path):
        print("dataset.csv not found, skipping...")
        return pd.DataFrame(columns=['resume_text', 'job_role'])

    df = pd.read_csv(path)
    df = df[['Resume', 'Role']].copy()
    df.columns = ['resume_text', 'job_role']

    # lowercase the role first to handle case inconsistencies
    # like "Data Scientist" vs "data scientist"
    df['job_role'] = df['job_role'].str.lower().str.strip()
    df['job_role'] = df['job_role'].map(DATASET_CSV_MAPPING)
    df = df.dropna(subset=['job_role', 'resume_text'])

    print(f"[dataset.csv] Loaded {len(df)} resumes")
    return df


def load_validation_csv(data_dir):
    """
    load validation.csv - this one is tricky because its a resume-job matching dataset
    the text column has resume and job description separated by [SEP]
    we extract just the resume part and use resume_domain as the role
    also only use positive matches (label=1) since those are confirmed pairs
    """
    path = os.path.join(data_dir, 'validation.csv')
    if not os.path.exists(path):
        print("validation.csv not found, skipping...")
        return pd.DataFrame(columns=['resume_text', 'job_role'])

    df = pd.read_csv(path)

    # only keep positive matches - where resume actually fits the role
    df = df[df['label'] == 1].copy()

    # extract just the resume part (before [SEP])
    df['resume_text'] = df['text'].apply(lambda x: str(x).split('[SEP]')[0].strip())

    # map resume_domain to unified roles
    df['job_role'] = df['resume_domain'].map(VALIDATION_CSV_MAPPING)
    df = df.dropna(subset=['job_role', 'resume_text'])

    # drop resumes that are too short (some might be garbage after splitting)
    df = df[df['resume_text'].str.len() > 50]

    print(f"[validation.csv] Loaded {len(df)} resumes")
    return df[['resume_text', 'job_role']]


def load_all_data(data_dir):
    """
    loads all 4 datasets and combines them into one big dataframe
    each loader normalizes columns to resume_text and job_role
    """
    print("=" * 50)
    print("LOADING ALL DATASETS")
    print("=" * 50)

    dfs = [
        load_original_dataset(data_dir),
        load_resume_csv(data_dir),
        load_dataset_csv(data_dir),
        load_validation_csv(data_dir),
    ]

    combined = pd.concat(dfs, ignore_index=True)

    # drop any rows with missing text or role
    combined = combined.dropna(subset=['resume_text', 'job_role'])
    combined = combined[combined['resume_text'].str.strip().astype(bool)]

    print(f"\nTotal combined: {len(combined)} resumes")
    print(f"\nJob role distribution in combined dataset:")
    print(combined['job_role'].value_counts())

    return combined


def train_and_evaluate():
    """
    main training function - trains 4 different models and picks the best one
    we compare multiple models because different algorithms work better
    for different types of data, and we want to find whats best for resumes
    """

    # load all datasets
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = load_all_data(data_dir)

    # preprocess all the text
    df = preprocess_dataset(df)

    # create TF-IDF features
    print("\n--- Creating TF-IDF features ---")
    X, vectorizer = create_tfidf_features(df['cleaned_text'])
    y = df['job_role']

    # save vectorizer right away so we dont forget
    vec_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
    save_vectorizer(vectorizer, vec_path)

    # split into train and test sets
    # using 80-20 split which is pretty standard
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # define the models we want to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
    }

    results = {}
    best_score = 0
    best_model_name = None
    best_model = None

    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # train the model
        model.fit(X_train, y_train)

        # predict on test set
        y_pred = model.predict(X_test)

        # calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # do cross validation too - gives a more reliable estimate
        # of how well the model generalizes
        # using 3 folds instead of 5 because the dataset is big now
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted')
        print(f"Cross-Val F1 (3-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # keep track of the best model
        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model = model

        # print detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # --- results summary ---
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(results).T
    print(results_df.round(4))

    print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")

    # save the best model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # --- generate plots ---
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 1. model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_comparison.png'), dpi=150)
    print(f"Model comparison chart saved to plots/model_comparison.png")
    plt.close()

    # 2. confusion matrix for the best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    labels = sorted(y.unique())

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=150)
    print(f"Confusion matrix saved to plots/confusion_matrix.png")
    plt.close()

    # 3. class distribution chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df['job_role'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Dataset - Job Role Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Job Role')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_distribution.png'), dpi=150)
    print(f"Class distribution chart saved to plots/class_distribution.png")
    plt.close()

    return best_model, vectorizer, results


if __name__ == "__main__":
    best_model, vectorizer, results = train_and_evaluate()
