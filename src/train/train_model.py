import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("src/data/cleaned_reviews.csv")
df.dropna(subset=['cleaned_review'], inplace=True)

class_counts = df['sentiments'].value_counts()
max_class_size = class_counts.max()
df_balanced = pd.concat([
    df[df.sentiments == label].sample(max_class_size, replace=True, random_state=42)
    for label in class_counts.index
])

X = df_balanced['cleaned_review']
y = df_balanced['sentiments']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    stop_words=None,
    min_df=1,
    sublinear_tf=True
)
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y_encoded, test_size=0.2, random_state=42)

# Logistic Regression
log_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}
grid_lr = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), log_params, cv=3, scoring='accuracy')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print("\nLogistic Regression")
print("Best params:", grid_lr.best_params_)
print(classification_report(y_test, best_lr.predict(X_test), target_names=le.classes_))

# SVM
svm_params = {
    'C': [0.1, 1, 10]
}
grid_svm = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), svm_params, cv=3, scoring='accuracy')
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print("\nSVM")
print("Best params:", grid_svm.best_params_)
print(classification_report(y_test, best_svm.predict(X_test), target_names=le.classes_))

# Naive Bayes(TF-IDF)
nb_params = {
    'alpha': [0.1, 0.5, 1.0]
}
grid_nb = GridSearchCV(MultinomialNB(), nb_params, cv=3, scoring='accuracy')
grid_nb.fit(X_train, y_train)
best_nb = grid_nb.best_estimator_
print("\nNaive Bayes (TF-IDF)")
print("Best params:", grid_nb.best_params_)
print(classification_report(y_test, best_nb.predict(X_test), target_names=le.classes_))

count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=30000, stop_words='english')
X_count = count_vectorizer.fit_transform(X)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_count, y_encoded, test_size=0.2, random_state=42)

nb_count = GridSearchCV(MultinomialNB(), nb_params, cv=3, scoring='accuracy')
nb_count.fit(Xc_train, yc_train)
print("\nNaive Bayes (CountVectorizer)")
print("Best params:", nb_count.best_params_)
print(classification_report(yc_test, nb_count.predict(Xc_test), target_names=le.classes_))

model_scores = {
    "Logistic Regression": accuracy_score(y_test, best_lr.predict(X_test)),
    "SVM": accuracy_score(y_test, best_svm.predict(X_test)),
    "Naive Bayes (TF-IDF)": accuracy_score(y_test, best_nb.predict(X_test)),
    "Naive Bayes (CountVectorizer)": accuracy_score(yc_test, nb_count.predict(Xc_test))
}

best_model_name = max(model_scores, key=model_scores.get)
print(f"\nBest performing model: {best_model_name} with accuracy = {model_scores[best_model_name]:.4f}")

if best_model_name == "Naive Bayes (CountVectorizer)":
    joblib.dump(nb_count.best_estimator_, "src/models/sentiment_model.pkl")
    joblib.dump(count_vectorizer, "src/models/vectorizer.pkl")
else:
    best_model = {
        "Logistic Regression": best_lr,
        "SVM": best_svm,
        "Naive Bayes (TF-IDF)": best_nb
    }[best_model_name]
    joblib.dump(best_model, "src/models/sentiment_model.pkl")
    joblib.dump(vectorizer, "src/models/vectorizer.pkl")

joblib.dump(le, "src/models/label_encoder.pkl")
print("Model, vectorizer, and label encoder saved successfully.")
