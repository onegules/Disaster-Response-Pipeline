# Disaster Response Pipeline

### Step 0: Set up a Conda Enviroment (Optional)

More information on how to do this is here:
https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

Given an anaconda enviroment, you need only install plotly. To do this run

'''python
conda install plotly
'''

in your console.
### Step 1: Set up the model and data

Run the following commands (in this order) in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### Step 2: Run

Run the following command in the app's directory to run the web app.
    `python run.py`

Then, go to http://127.0.0.1:3001/ to access it on your local computer.
