import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_data(data):
    
    if not os.path.exists(data):
        print("Error: file does not exist.")
        return None
    df = pd.read_csv(data)
    
    return df

def preprocess_data (df, min_population=100000):

    df = df[['city', 'state_id', 'population', 'lat', 'lng']].copy()
    df = df.dropna()
    df = df[df['population'] >= min_population]
    df = df.drop_duplicates(subset=['city', 'state_id'])
    df = df.reset_index(drop=True)
    df.insert(0, 'city_id', range(len(df)))
    
    return df