import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Takes paths of two datasets as input, merges their dataframe 
    and returns the resulting dataframe
    
    '''

    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df=messages.merge(categories, on="id")
    return df

def clean_data(df):
    '''
    splits the categories column into separate columns,
    converts values to binary, and drops duplicates
        Input
            df: dataframe to be cleaned

        Output
            df: cleaned dataframe    
    '''
    categories = df.categories.str.split(";",expand=True)
    
    row = categories.iloc[0]

    category_colnames = row.apply(lambda x:x[:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
      
        categories[column] = categories[column].str[-1]

        categories[column] = categories[column].astype("int")


    df.drop("categories",axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):    
    '''
    saves the dataframe in a sqlite database in a table "msg_cat"
    '''
    engine = create_engine(f'sqlite:///{database_filename.db}')
    df.to_sql('msg_cat', engine, index=False)


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