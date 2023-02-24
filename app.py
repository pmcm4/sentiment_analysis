from flask import Flask, request, render_template
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

CSV_FILE_PATH = 'tagalog_data.csv'

try:
    with open(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

df.fillna('', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Define the pipeline and hyperparameters to search
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC())])
parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
              'tfidf__use_idf': (True, False),
              'clf__C': [0.1, 1, 10, 100],
              'clf__kernel': ['linear', 'rbf', 'poly']}

# Search for the best hyperparameters using Grid Search
clf = GridSearchCV(text_clf, parameters, cv=5)
clf.fit(X_train, y_train)

# Use the trained model to predict the sentiment of the test set
y_pred = clf.predict(X_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        text = request.form.get('text')
        sentiment = int(clf.predict([text])[0])
        result = sentiment
        return {'result': result}
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 3000)),host='0.0.0.0',debug=True)
