import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download required nltk data (only needs to run once but doesnt hurt to call again)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Main preprocessing function - takes raw resume text and cleans it up
    for the classifier. We do this step by step so its easier to debug
    if something goes wrong.
    """

    # convert to lowercase first
    text = text.lower()

    # remove urls - resumes often have linkedin links and portfolio urls
    # that just add noise to the model
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # remove phone numbers (different formats people use)
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', '', text)

    # get rid of punctuation and special characters
    # keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra whitespace that builds up after all the regex stuff
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize into words
    tokens = word_tokenize(text)

    # removing stopwords here because resumes contain a lot of filler words
    # like "the", "and", etc which don't help the classifier much
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize - this converts words to their base form
    # eg "programming" -> "programming" (stays same), "experienced" -> "experienced"
    # it helps reduce the vocabulary size a bit
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # filter out really short words (1-2 chars) - they're usually not meaningful
    tokens = [word for word in tokens if len(word) > 2]

    # join back into a single string
    cleaned = ' '.join(tokens)
    return cleaned


def preprocess_dataset(df):
    """
    applies cleaning to the whole dataframe
    the resume_text column gets cleaned and stored in a new column
    so we can always go back to the original if needed
    """
    print("Preprocessing resumes...")
    df['cleaned_text'] = df['resume_text'].apply(clean_text)

    # show a quick sample to make sure it looks right
    print(f"Processed {len(df)} resumes")
    print(f"\nSample original: {df['resume_text'].iloc[0][:100]}...")
    print(f"Sample cleaned:  {df['cleaned_text'].iloc[0][:100]}...")

    return df


if __name__ == "__main__":
    # quick test with a sample text
    sample = "John Doe | john@email.com | (555) 123-4567\nExperienced Software Engineer with 5+ years in Python, Java, and Cloud Computing. Built scalable REST APIs."
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
