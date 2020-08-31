import pandas as pd
import numpy as np

def execute(input_file, output_file):
    '''
    Data preprocessing stage: Deleting the unwanted anamolies
    '''
    
    data = pd.read_csv('C:/Users/Akshay/train.csv', sep = ";")
    
    '''
    Deleting or dropping out the null values
    '''
    data = data.dropna()
    '''
    Deleting the unwanted columns as it is mainly consists of strings
    '''
    data = data.drop(["Name","Ticket", "Cabin"], axis = 1, inplace = True)

    data.to_csv(output_file)

execute('train.csv', 'testrain.csv')
