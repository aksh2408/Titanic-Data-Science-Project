import pandas as pd
import numpy as np
import sklearn

def execute(input_file, output_file, force_write = True):
    """Builds features:
        This step is Data Cleanzing

    Args:
        input_file (str): input file.
    """

    '''
    Data preprocessing stage: Deleting the unwanted anamolies
    '''
    
    df = pd.read_csv('C:/Users/Akshay/train.csv', sep = ";")
    
    df = df.dropna() #Deleting or dropping out the null values
    
    df = df.drop(["Name","Ticket", "Cabin"], axis = 1, inplace = True) 
    #Deleting the unwanted columns as it is mainly consist of strings
    
    df["Sex"] = df["Sex"].replace("male", 0)
    df["Sex"] = df["Sex"].replace("female", 1)

    sex_mapping = {'male': 0, 'female': 1}
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map[sex_mapping]
        
        
    embarked_dict = {}
    embarked_dict_values = 0
    for i in df.Embarked:
        if i in embarked_dict.keys():
            pass
        else:
            embarked_dict_values = embarked_dict_values + 1
            embarked_dict[i] = embarked_dict_values
    
    for i in embarked_dict.keys():
        df["Embarked"].replace(i, embarked_dict[i], inplace = True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    df.to_csv(output_file)

execute('train.csv', 'subtrain.csv')