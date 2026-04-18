import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already done
nltk.download('stopwords')

# Load the dataset (adjust path if needed)
df = pd.read_csv('../../Datasets/spam.csv', encoding='latin-1')  # Common encoding for this dataset

# Assuming columns: 'v1' for label (ham/spam) and 'v2' for message
# Rename for clarity
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

# Preprocess text: remove punctuation, lowercase, remove stopwords
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)

# Encode labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into features (X) and labels (y)
X = df['message']
y = df['label']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
X_vectorized = vectorizer.fit_transform(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict on test set
y_pred = nb_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
