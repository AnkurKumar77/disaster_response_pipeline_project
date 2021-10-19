# Disaster Response Pipeline Project


### Project Summary

In this project, data from "Figure Eight" is used which includes the disaster messages sent during real disaster events. The task is to build and train a classifier which can classify new messages to approriate categories. A web app is also created where a person can just enter a message and it will be classified into appropriate categories.


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
  
### Web App

Here are some screenshots of the web app and the classification results:

![home](https://user-images.githubusercontent.com/89351216/137963981-e60fb03b-9e46-4bb5-8ea4-0453da781cb7.PNG)

![out](https://user-images.githubusercontent.com/89351216/137964018-495bf191-4fce-48a2-bb7f-a617c83af6ae.PNG)

### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Responses.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Responses.db models/clf.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


