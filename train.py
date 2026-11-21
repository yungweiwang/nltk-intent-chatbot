
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_text

# 1. Load dataset
with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# 2. Split data before vectorization
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# 3. Bag of Words
vectorizer = CountVectorizer(
    tokenizer=preprocess_text,
    token_pattern=None,
    lowercase=False
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 4. Train Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

dt_model.fit(X_train, y_train)

# 5. Evaluate
y_pred = dt_model.predict(X_test)

print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# 6. Test a new sentence
example = ["Hello there!"]
example_vector = vectorizer.transform(example)

prediction = dt_model.predict(example_vector)

print("Example:", example[0])
print("Predicted intent:", prediction[0])


from sklearn.naive_bayes import MultinomialNB

# 7. Train Naive Bayes
nb_model = MultinomialNB(alpha=1.0)

nb_model.fit(X_train, y_train)

# 8. Evaluate Naive Bayes
nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)
dt_accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Accuracy:", nb_accuracy)

print("\nNaive Bayes Classification Report:")
print(classification_report(
    y_test,
    nb_pred,
    zero_division=0
))

# 9. Compare models
print("\nModel Comparison")
print(f"Decision Tree: {dt_accuracy:.2%}")
print(f"Naive Bayes:   {nb_accuracy:.2%}")

# 10. Test new sentences
examples = [
    "Hello there!",
    "Can you explain NLP?",
    "I want to learn Python",
    "Thank you for helping me"
]

print("\nExample Predictions:")

for sentence in examples:
    vector = vectorizer.transform([sentence])

    dt_prediction = dt_model.predict(vector)[0]
    nb_prediction = nb_model.predict(vector)[0]

    print(f"\nInput: {sentence}")
    print(f"Decision Tree: {dt_prediction}")
    print(f"Naive Bayes:   {nb_prediction}")


import os
import pickle

# Create models directory
os.makedirs("models", exist_ok=True)

# Train the final model using all 48 examples
final_vectorizer = CountVectorizer(
    tokenizer=preprocess_text,
    token_pattern=None,
    lowercase=False
)

X_full = final_vectorizer.fit_transform(texts)

final_model = MultinomialNB(alpha=1.0)
final_model.fit(X_full, labels)

# Save model and vectorizer together
with open("models/chatbot_model.pkl", "wb") as file:
    pickle.dump(
        {
            "model": final_model,
            "vectorizer": final_vectorizer
        },
        file
    )

print("\nFinal chatbot model saved successfully!")

