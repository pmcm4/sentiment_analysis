from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

df = pd.read_csv('tagalog_data.csv', on_bad_lines='skip')
df.fillna('', inplace=True)
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
text_clf.fit(df['text'], df['sentiment'])
df.dropna(inplace=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        text = request.form.get('text')
        sentiment = int(text_clf.predict([text])[0])
        result = sentiment
        return {'result': result}
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
