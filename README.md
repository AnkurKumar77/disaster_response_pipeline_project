# Disaster Response Pipeline Project


### Project Summary

In this project, data from "Figure Eight" is used which includes the disaster messages sent dring real disaster events. The task is to build and train a classifier which can classify new messages to approriate categories. A web app is also created where a person can just enter a message and it will be classified into appropriate categories.


### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # ETL code    


models   

|- train_classifier.py # Building and training model code    
  


### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Responses.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Responses.db models/clf.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
