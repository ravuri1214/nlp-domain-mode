from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data import texts, labels


def train_for_inference():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer


def predict_label(text: str):
    model, vectorizer = train_for_inference()
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]


if __name__ == "__main__":
    sample_text = "customer unable to login"
    predicted_label = predict_label(sample_text)
    print(f"Input: {sample_text}")
    print(f"Predicted Label: {predicted_label}")
