import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Extracts data from the SQLite database

    returns:

        input Variable,
        target variable,
        category names

    '''
    
    engine = create_engine(f'sqlite:///{database_filepath.db}')
    df = pd.read_sql_table("msg_cat",engine)
    
    X = df.message    
    Y =df.iloc[:,4:] 
    
    category_names=list(Y.columns)
    Y.related.replace(2,1,inplace=True)
    
    return X, Y, category_names

def tokenize(text):
    '''
    Applies case normalization, lemmatization, and tokenization to text.

    Input:

        text: raw text

    Output:
        tokens: cleaned tokens list   
    '''
    
    lemmatizer=WordNetLemmatizer()
    text=re.sub(r"^[a-zA-Z0-9]"," ",text.lower())
    tokens=word_tokenize(text)
    tokens=[lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")]
    return tokens
    
    


def build_model():
    '''
    Builds a pipeline for processing text and performing multi-output classification 
    Applies GridSearchCV for fine tuning

    Output:

        cv: model
    '''
    
    pipeline = Pipeline([
    ("vect",CountVectorizer(tokenizer=tokenize)),
    ("tfidf",TfidfTransformer()),
    ("clf",MultiOutputClassifier(ExtraTreesClassifier()))
])
    parameters = {
    "vect__max_df":(0.75,1.0),
    "tfidf__use_idf":(True,False)
}

    cv = GridSearchCV(pipeline,parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluates the trained model and displays the precision, recall, f1-score for the model

    '''
    y_pred=model.predict(X_test)
    y_pred=pd.DataFrame(y_pred,columns=category_names)
    for col in category_names:
        print(col.upper())
        print(classification_report(Y_test.loc[:,col],y_pred.loc[:,col]))


def save_model(model, model_filepath):
    '''
        Exports the trained model in a pkl file
    '''

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)




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