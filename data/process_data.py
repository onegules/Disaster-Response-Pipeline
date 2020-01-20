import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - (str) The path to the disaster_messages.csv file
    categories_filepath - (str) The path to the disaster_categories.csv file

    OUTPUT:
    df - (pandas dataframe) The dataframe given by the merged csv files
    '''
    # Load the two csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge into df
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    INPUT:
    df - (pandas dataframe) df as defined in load_data

    OUTPUT:
    df - (pandas dataframe) df after it has been cleaned

    Description:
    Takes df and its categories column and splits the values into their own
    columns and uses the information in the column to place a 1 or a 0
    if the message (row) is part of that category
    '''
    categories = df['categories'].str.split(';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Get and save the category names to be made into column names
    category_colnames = []
    for words in row:
        category_colnames.append(words[0:len(words)-2])

    # Create the new columns with 0 or 1 in the corresponding rows
    categories.columns = category_colnames
    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Remove rows that are not 0 or 1 in a given category
    faulty_data_df = categories[categories['related'] == 2]
    if(faulty_data_df.size != 0):
        categories = categories[categories.related != 2]
    else:
        pass

    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    df.drop('original', axis=1, inplace=True)

    # Add the new categories columns to df
    df = pd.concat([df,categories], axis=1)

    # Drop any duplicates or NaN valuues
    df.drop_duplicates(inplace=True)
    df = df.dropna()

    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - (pandas dataframe) df as defined in clean_data
    database_filename - (str) The name to save the database as

    OUTPUT:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
