import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt_tab")
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords, then stem
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]
    return " ".join(tokens)

with open("data.json", "r") as file:
    data = json.load(file)

stories = []
questions = []
answers = []

for item in data["data"]:
    story = item["story"]
    for q, a in zip(item["questions"], item["answers"]):
        stories.append(story)
        questions.append(q["input_text"])
        answers.append(a["input_text"])
        if "additional_answers" in item:
            for add_answers in item["additional_answers"].values():
                for add_a in add_answers:
                    if (
                        add_a["turn_id"] == a["turn_id"]
                        and add_a["input_text"] != a["input_text"]
                    ):
                        stories.append(story)
                        questions.append(q["input_text"])
                        answers.append(add_a["input_text"])

df = pd.DataFrame({"story": stories, "question": questions, "answer": answers})

df["processed_story"] = df["story"].apply(preprocess_text)
df["processed_question"] = df["question"].apply(preprocess_text)
df["processed_answer"] = df["answer"].apply(preprocess_text)

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)

story_features = tfidf_vectorizer.fit_transform(df["processed_story"])
question_features = tfidf_vectorizer.transform(df["processed_question"])

# Create a feature matrix by concatenating story and question features
# feature_matrix = pd.concat(
#     [
#         pd.DataFrame(
#             story_features.toarray(),
#             columns=[f"story_feat_{i}" for i in range(story_features.shape[1])],
#         ),
#         pd.DataFrame(
#             question_features.toarray(),
#             columns=[f"question_feat_{i}" for i in range(question_features.shape[1])],
#         ),
#     ],
#     axis=1,
# )

# Add the preprocessed answer column to the feature matrix
# feature_matrix["processed_answer"] = df["processed_answer"]

# Save the preprocessed data and feature matrix
df.to_csv("preprocessed_data.csv", index=False)
# feature_matrix.to_csv("feature_matrix.csv", index=False)

print("Data preprocessing and feature extraction completed.")
print(f"Preprocessed data shape: {df.shape}")
# print(f"Feature matrix shape: {feature_matrix.shape}")
