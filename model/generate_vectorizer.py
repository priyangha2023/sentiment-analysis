import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the 'model' directory exists
os.makedirs("model", exist_ok=True)

# Example text data for training the vectorizer
sample_texts = ["I love this product!", "This is terrible.", "Okay, not great but not bad."]

# Train a new vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(sample_texts)

# Save the vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Vectorizer saved successfully in 'model/vectorizer.pkl'!")
