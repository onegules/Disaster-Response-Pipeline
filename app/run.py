from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'stopwords', 'wordnet'])

app = Flask(__name__)


def tokenize(text):
    '''
    INPUT:
    text - (str) The text to tokenize

    OUTPUT:
    clean_tokens - (list) A tokenized list of the text after normalizing it
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/DisasterModel")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Get related and unrelated dataframes
    related = df[df['related'] == 1]
    unrelated = df[df['related'] == 0]

    # Get top 5 category types not including related
    top_sorted_list = []
    for column in related.columns[4:]:
        name_sum_pair = column, related[column].sum()
        top_sorted_list.append(name_sum_pair)

    top_sorted_list.sort(key=lambda x: x[1], reverse=True)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': '',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=['related', 'unrelated'],
                    y=[related.shape[0], unrelated.shape[0]]
                )
            ],

            'layout': {
                'title': 'Distribution of Relation Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type Of Message"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[x[0] for x in top_sorted_list][:5],
                    y=[y[1] for y in top_sorted_list][:5]
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories of those Related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Of Message"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
