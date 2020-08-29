from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def home_post():
    stop_words = stopwords.words('english') #Getting the stop words from nltk
    review = request.form['review'].lower() #Getting the review from home.html form

    #Removing the stop words from the review
    processed_review = ' '.join([word for word in review.split() if word not in stop_words]) 

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_review)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('home.html', final=compound, review=review)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)