# NLTK Intent Classification Chatbot

A Python-based intent classification chatbot built using **NLTK** and **Scikit-learn**. This project demonstrates an end-to-end traditional NLP pipeline, from text preprocessing to machine learning classification and interactive chatbot responses.

## Features

- Tokenization using NLTK
- Stopword removal and POS tagging
- Lemmatization and stemming
- Bag-of-Words feature extraction
- Decision Tree and Multinomial Naive Bayes classification
- Intent prediction with a confidence-based fallback
- Interactive command-line chatbot

## NLP Pipeline

```text
User Input
    ↓
Tokenization
    ↓
POS Tagging
    ↓
Stopword Removal
    ↓
Lemmatization
    ↓
Bag of Words
    ↓
Naive Bayes Classifier
    ↓
Intent Prediction
    ↓
Chatbot Response
```

## Dataset

The chatbot uses a custom JSON dataset containing 48 example utterances across six intent categories:

- Greeting
- Goodbye
- Thanks
- Natural Language Processing
- Machine Learning
- Python

## Model Performance

Models were evaluated using a stratified 75/25 train-test split with `random_state=42`.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 66.67% | 0.64 |
| Multinomial Naive Bayes | 75.00% | 0.71 |

Naive Bayes achieved higher accuracy on this particular test split and was selected for the chatbot engine.

**Limitations:** The dataset contains only 48 examples, with 12 held-out test examples. These results are preliminary and should not be interpreted as evidence of robust generalization. Both classifiers struggled with greeting recognition during evaluation.

The final deployed model was retrained on all 48 examples.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nltk-intent-chatbot.git
cd nltk-intent-chatbot
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Train and save the model:

```bash
python train.py
```

Run the chatbot:

```bash
python chatbot.py
```

Example interaction:

```text
You: What is NLP?
Chatbot: NLP enables computers to process human language.

You: Tell me about Python
Chatbot: Python is a programming language widely used in NLP and machine learning.

You: Thanks
Chatbot: Happy to help!

You: Hello there!
Chatbot: I'm not sure I understand. Could you rephrase?

You: quit
Chatbot: Goodbye!
```

## Project Structure

```text
nltk-intent-chatbot/
├── data/
│   └── intents.json
├── models/
│   └── chatbot_model.pkl
├── preprocessing.py
├── train.py
├── chatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations and Future Improvements

- Expand the training dataset with more diverse utterances.
- Evaluate performance using repeated stratified cross-validation.
- Improve recognition of greeting and out-of-domain inputs.
- Tune the fallback threshold using validation data.
- Compare Bag of Words against TF-IDF and contextual embeddings.
- Explore transformer-based intent classification.

## Technologies

Python, NLTK, Scikit-learn, NumPy, JSON, Git

## License

This project is intended for educational and portfolio purposes.