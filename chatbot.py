
import json
import pickle
import random

from preprocessing import preprocess_text

# Load trained model
with open("models/chatbot_model.pkl", "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
vectorizer = saved["vectorizer"]

# Load chatbot responses
with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

responses = {
    intent["tag"]: intent["responses"]
    for intent in data["intents"]
}


def chatbot_response(message, threshold=0.45):
    """Predict intent and return a chatbot response."""

    # Preprocess and vectorize input
    vector = vectorizer.transform([message])

    # Handle unknown vocabulary
    if vector.nnz == 0:
        return "Sorry, I don't understand. Could you rephrase?"

    # Predict intent probabilities
    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max()
    predicted_intent = model.classes_[probabilities.argmax()]

    # Reject low-confidence predictions
    if confidence < threshold:
        return "I'm not sure I understand. Could you rephrase?"

    # Select a response
    return random.choice(responses[predicted_intent])


if __name__ == "__main__":
    print("Chatbot: Hello! Type 'quit' to exit.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break

        if not user_input:
            continue

        response = chatbot_response(user_input)
        print("Chatbot:", response)

