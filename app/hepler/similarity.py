import re
from math import exp
from sklearn.metrics.pairwise import cosine_similarity


def contains_word(word, text):
    pattern = rf"\b{word}\b"
    return re.search(pattern, text.lower()) is not None


def extract_words(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def compute_cosine_similarity(vector, other_vectors):
    return cosine_similarity([vector], other_vectors)[0]


def normalize_scores(score, a=0, b=1, mu=0.5, k=2):
    sigmoid_score = 1 / (1 + exp(-k * (score - mu)))
    # Chuẩn hóa giá trị vào khoảng [a, b]
    normalized_score = a + (b - a) * sigmoid_score

    return normalized_score
