import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import nltk
nltk.download(['punkt','stopwords'])
import pickle

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath -  (str) The path to the database

    OUTPUT:
    X - (numpy array) An array of the messages column in the database
    Y - (pandas dataframe) The columns corresponding to each message in database
    columns - (list) A list of the columns of df
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df.iloc[:,3:]
    columns = df.columns

    return X, Y, columns


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [words for words in tokens if words not in stopwords.words("english")]
    return tokens



def build_model():
    # Create pipeline object
    pipeline = Pipeline([
        ('vect', HashingVectorizer(tokenizer=tokenize,n_features=2**4)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Create a list of parameters to grid search
    parameters = {'tfidf__smooth_idf':(True, False)}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    # Get the values predicted
    y_pred = model.predict(X_test)
    predicted_categories = category_names[3:]

    # Print out the evaluation results
    print('Note: The order of the results are [(For 0), (For 1)]\n')
    for i in range(len(predicted_categories)):
        print('For {}: \n'.format(predicted_categories[i]))
        precision,recall,fscore,support = score(Y_test.iloc[:,i],y_pred[:,i])
        print('f1-score: {}'.format(fscore))
        print('precision: {}'.format(precision))
        print('recall: {} \n\n'.format(recall))


def save_model(model, model_filepath):
    with open('DisasterModel', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
