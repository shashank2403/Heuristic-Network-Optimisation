import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_data(data : str) -> pd.DataFrame:
    
    if not os.path.exists(data):
        print("Error: file does not exist.")
        return None
    df = pd.read_csv(data)
    
    return df

def preprocess_data (df: pd.DataFrame, min_population : float =1e5) -> pd.DataFrame:

    df = df[['city', 'population', 'country', 'lat', 'lng']].copy()
    df = df.dropna()
    df = df[df['country'] == 'India']
    df = df[df['population'] >= min_population]
    df = df[df['lng'].between(65, 100) & df['lat'].between(5, 40)]
    df = df.drop_duplicates(subset=['city'])
    df = df.reset_index(drop=True)
    df.insert(0, 'city_id', range(len(df)))
    
    return df

def plot_cities(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot all cities
    ax.scatter(df['lng'], df['lat'], s=30, alpha=0.5, c='steelblue', label='All cities')

    ax.set_xlim(65, 100)
    ax.set_ylim(5, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('data/cities_graph.png', dpi=200)
    plt.show()