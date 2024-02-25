from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

comments_data = []  # to store comments temporarily, replace with a database in production

# Load model and tokenizer for sentiment analysis
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(roberta)
sentiment_tokenizer = AutoTokenizer.from_pretrained(roberta)
sentiment_labels = ['Negative', 'Neutral', 'Positive']

@app.route('/')
def index():
    return render_template('community.html', comments=comments_data)

@app.route('/add_comment', methods=['POST'])
def add_comment():
    name = request.form.get('name')
    comment_text = request.form.get('comment')

    # Perform sentiment analysis
    sentiment = analyze_sentiment(comment_text)

    # Store the comment and sentiment in the temporary list
    comments_data.append({'name': name, 'comment': comment_text, 'sentiment': sentiment})

    # Save comments to an Excel file
    save_comments_to_excel()

    return redirect(url_for('index'))

def analyze_sentiment(text):
    # Preprocess text
    tweet_words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)

    # Sentiment analysis
    encoded_tweet = sentiment_tokenizer(tweet_proc, return_tensors='pt')
    output = sentiment_model(**encoded_tweet)

    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    # Get the predicted sentiment label
    sentiment_index = scores.argmax()
    sentiment = sentiment_labels[sentiment_index]

    return sentiment

def save_comments_to_excel():
    # Convert the comments_data list to a DataFrame
    df = pd.DataFrame(comments_data)

    # Save the DataFrame to an Excel file
    df.to_excel(r'C:\Users\HP\Desktop\byte\FLASKKK\comments_data.xlsx', index=False)
    


if __name__ == '__main__':
    app.run(debug=True)
