from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from data import texts, labels


def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return model, vectorizer


if __name__ == "__main__":
    train_model()
