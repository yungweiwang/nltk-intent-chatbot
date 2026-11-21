import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required resources
for resource in [
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4"
]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preserve important negation words
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "not", "nor"}


def get_wordnet_pos(tag):
    """Convert NLTK POS tags to WordNet POS tags."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text, use_stemming=False):
    """Convert raw text into normalized tokens."""

    # 1. Tokenization
    tokens = word_tokenize(text.lower())

    # 2. POS tagging (before removing stopwords)
    pos_tags = nltk.pos_tag(tokens)

    # 3. Remove punctuation and stopwords
    filtered = [
        (word, pos)
        for word, pos in pos_tags
        if word.isalpha() and word not in stop_words
    ]

    # 4. Lemmatization or stemming
    if use_stemming:
        return [
            stemmer.stem(word)
            for word, pos in filtered
        ]

    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in filtered
    ]


if __name__ == "__main__":
    text = "The students are studying natural language processing."

    print("Original:", text)
    print("Lemmatization:", preprocess_text(text))
    print("Stemming:", preprocess_text(text, use_stemming=True))


