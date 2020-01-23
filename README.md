# Disaster Response Pipeline

## The project

This project uses disaster data from [Figure Eight](https://www.figure-eight.com/),
cleans it and analyzes the messages coming in from users with natural language
processing techniques to train an ML model. This ML model is then used in a
flask web app ready for deployment. The web app can be used to take in new
disaster messages and the ML model will tell you what categories the message is
in.

## The files

### The app folder

In the app folder you'll find run.py and the templates folder. The templates folder
contains all the HTML information for the webpages. While run.py contains all
information corresponding to the flask app (the backend). The run.py file also
contains code for generating the graphics seen on the home page.

This folder also contains a screenshot of what the disaster response web app
should look like.

### The data folder

The data folder contains all the data required to be able to run the project.
That is, disaster_messages.csv and disaster_categories.csv. Both of these
are used in process_data.py to be combined and processed. The file process.py
returns a DisasterResponse.db file that is cleaned and ready to be used in
the train_classifier.py file in the models folder.

###  The models folder

Before running anything, the models folder contains only train_classifier.py.
This trains the ML model and saves the file to a pickle file. This file is called
DisasterModel and will appear on running the application. DisasterModel is then
used in the backend in the application folder to give the results of a new message.

## Follow the steps to run the web application


### Step 0: Set up a Conda Enviroment (Optional)

More information on how to do this is here:
https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

Given an anaconda enviroment, you need only install plotly. To do this run

```python
conda install plotly
```

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


![Screenshot](/app/screenshot.PNG)
