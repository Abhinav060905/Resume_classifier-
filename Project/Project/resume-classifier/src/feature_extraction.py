from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# TF-IDF helps find words that are important in one resume
# but not common across all resumes
# for example "tensorflow" might appear in a data scientist's resume
# but not in a business analyst's resume - so it becomes a strong signal


def create_tfidf_features(texts, max_features=5000):
    """
    converts cleaned text into TF-IDF feature vectors

    max_features=5000 means we only keep the top 5000 most important words
    this keeps things manageable and avoids overfitting on rare words
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # using unigrams and bigrams
                               # bigrams help capture phrases like "machine learning"
                               # instead of treating "machine" and "learning" separately
        min_df=2,             # ignore words that appear in less than 2 resumes
        max_df=0.95           # ignore words that appear in 95%+ of resumes (too common)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features (unique words/phrases): {len(vectorizer.get_feature_names_out())}")

    # show top features just to sanity check
    feature_names = vectorizer.get_feature_names_out()
    print(f"Sample features: {list(feature_names[:10])}")

    return tfidf_matrix, vectorizer


def transform_text(text, vectorizer):
    """
    use an already fitted vectorizer to transform new text
    this is what we use during prediction - we cant fit again
    because it would change the feature space
    """
    return vectorizer.transform([text])


def save_vectorizer(vectorizer, path='models/vectorizer.pkl'):
    """save the fitted vectorizer so we can reuse it later"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to {path}")


def load_vectorizer(path='models/vectorizer.pkl'):
    """load a previously saved vectorizer"""
    return joblib.load(path)


if __name__ == "__main__":
    # quick test
    sample_texts = [
        "python machine learning tensorflow data science",
        "java spring boot microservices backend api",
        "html css javascript react frontend web"
    ]
    matrix, vec = create_tfidf_features(sample_texts, max_features=100)
    print(f"\nTest matrix shape: {matrix.shape}")
