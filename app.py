import heapq
import re
import nltk
from flask import Flask, render_template, request

# pip install spacy
import spacy

app = Flask(__name__)

# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')


def summarize_text(text, num_sentences=5):
    # Remove citations and extra white space
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove non-letter characters and extra white space
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    # Tokenize sentences using SpaCy
    sentence_list = [sent.text.strip() for sent in nlp(text).sents]

    # Remove stop words and calculate word frequencies
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_text):
        if word.lower() not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize word frequencies
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # Select top sentences based on sentence scores
    summary_sentences = heapq.nlargest(
        num_sentences, sentence_scores, key=sentence_scores.get)

    # Join selected sentences into a summary
    summary = ' '.join(summary_sentences)
    return summary


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', error='Please enter some text.')
        try:
            num_sentences = int(request.form['num_sentences'])
        except ValueError:
            num_sentences = 5
        summary = summarize_text(text, num_sentences)
        return render_template('index.html', summary=summary)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
