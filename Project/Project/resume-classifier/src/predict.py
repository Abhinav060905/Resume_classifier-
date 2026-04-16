import sys
import os
import joblib
import numpy as np

# add parent dir to path so we can import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess import clean_text
from utils.pdf_parser import extract_text_from_pdf


def load_model_and_vectorizer():
    """load the saved model and vectorizer from the models folder"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    model_path = os.path.join(model_dir, 'classifier.pkl')
    vec_path = os.path.join(model_dir, 'vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("Error: Model files not found! Run train_model.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    return model, vectorizer


def predict_role(text, model, vectorizer):
    """
    takes resume text and returns the predicted job role
    also returns confidence scores if the model supports it
    """
    # preprocess the text same way we did during training
    cleaned = clean_text(text)

    # transform using the saved vectorizer
    features = vectorizer.transform([cleaned])

    # predict
    prediction = model.predict(features)[0]

    # try to get confidence/probability scores
    # not all models support predict_proba (like LinearSVC)
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba)
        # also get top 3 predictions for more detail
        classes = model.classes_
        top_indices = np.argsort(proba)[::-1][:3]
        top_predictions = [(classes[i], round(proba[i], 4)) for i in top_indices]
    else:
        # for models like SVM that use decision_function instead
        top_predictions = [(prediction, "N/A")]

    return prediction, confidence, top_predictions


def predict_from_pdf(pdf_path):
    """
    end-to-end prediction from a PDF file
    extracts text -> preprocesses -> predicts
    """
    print(f"Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("Could not extract text from PDF. The file might be scanned/image-based.")
        return None

    print(f"Extracted {len(text)} characters from PDF")

    model, vectorizer = load_model_and_vectorizer()
    prediction, confidence, top_preds = predict_role(text, model, vectorizer)

    print("\n" + "="*40)
    print(f"Predicted Role: {prediction}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2f}")
    print("\nTop predictions:")
    for role, score in top_preds:
        print(f"  {role}: {score}")
    print("="*40)

    return prediction


def predict_from_text(text):
    """predict from raw text input (for testing without a PDF)"""
    model, vectorizer = load_model_and_vectorizer()
    prediction, confidence, top_preds = predict_role(text, model, vectorizer)

    print("\n" + "="*40)
    print(f"Predicted Role: {prediction}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2f}")
    print("\nTop predictions:")
    for role, score in top_preds:
        print(f"  {role}: {score}")
    print("="*40)

    return prediction


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if file_path.endswith('.pdf'):
            predict_from_pdf(file_path)
        else:
            # treat it as a text file
            with open(file_path, 'r') as f:
                text = f.read()
            predict_from_text(text)
    else:
        # demo with a sample resume text
        sample = """
        Experienced data scientist with 3 years of experience in machine learning
        and deep learning. Proficient in Python, TensorFlow, scikit-learn, and pandas.
        Built recommendation systems and NLP models. Strong background in statistics
        and data visualization. Masters degree in Computer Science.
        """
        print("Running demo prediction with sample text...")
        predict_from_text(sample)
