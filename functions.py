import numpy as np 
import pandas as pd 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.preprocessing import OrdinalEncoder

def get_data(years):
    # Get data from all races of predefined seasons
    import fastf1
    races = []
    # years = [2022,2023,2024,2025]
    for year in years:
        print(year)
        event_schedule = fastf1.get_event_schedule(year,include_testing=False,backend='fastf1')
        no_races = len(event_schedule["EventName"])

        for race in range(1,no_races):
            session = fastf1.get_session(year, race, 'R')
            if year == 2025 and race >= 6:
                print("Miami Grand Prix 2025 is not available, now looking into the future")
            else:
                session.load()
                try:
                    weather = session.weather_data[["AirTemp","Rainfall"]]
                except:
                    print("Weather data not available")
                    continue
                location = session.event["EventName"]
                race = session.results[["Abbreviation","TeamName","GridPosition","Position"]]
                race["EventName"] = location
                race["Year"] = year
                try:
                    race["Rainfall"] = weather["Rainfall"].any() # Was there any rain in the session
                    race["AirTemp"] = weather["AirTemp"].mean() # Average air temperature in the session
                except:
                    race["Rainfall"] = False
                    race["AirTemp"] = 22.0 # Default air temperature in case of missing data
                races.append(race)
        

    races = pd.concat(races, ignore_index=True)
    return races

def preprocess_data(races):
    # Make feature table
    data = races.iloc[:,[3,4,0,1,2,5,6,7]].dropna()
    # print(data.tail(20))

    # New feature: driver performance
    data["AveragePos"] = data.groupby("Abbreviation")["Position"].shift(1).rolling(window=3,min_periods=1).mean().reset_index(drop=True)

    # Split features and target
    X = data[["Abbreviation","TeamName","EventName","Year","Rainfall","AirTemp","GridPosition","AveragePos"]]
    Y = data[["Position"]]

    # Label encoding for XGB model
    label_encoder_abb = preprocessing.LabelEncoder()
    label_encoder_abb.fit(X["Abbreviation"])
    label_encoder_team = preprocessing.LabelEncoder()
    label_encoder_team.fit(X["TeamName"])
    label_encoder_event = preprocessing.LabelEncoder()
    label_encoder_event.fit(X["EventName"])
    label_encoder_rain = preprocessing.LabelEncoder()
    label_encoder_rain.fit(X["Rainfall"])

    X_le = pd.DataFrame()
    X_le["Abbreviation"] = label_encoder_abb.transform(X["Abbreviation"])
    X_le["TeamName"] = label_encoder_team.transform(X["TeamName"])
    X_le["EventName"] = label_encoder_event.transform(X["EventName"])
    X_le["Year"] = X["Year"]
    X_le["Rainfall"] = label_encoder_rain.transform(X["Rainfall"])
    X_le["AirTemp"] = X["AirTemp"]
    X_le["GridPosition"] = X["GridPosition"]
    X_le["AveragePos"] = X["AveragePos"]

    return X_le, Y, label_encoder_abb, label_encoder_team, label_encoder_event, label_encoder_rain

def train_test_split(X_le,Y):
    # Train & test split
    train_split = int(len(X_le)*.70) #How many data points is 70% n--> used for training
    test_split = int(len(X_le)-train_split)
    test_split_sample = len(X_le) - test_split
    # print(X.iloc[range(test_split_sample-10,test_split_sample+10),:]) # split exactly between two races
    test_split_sample = 678

    X_train = X_le.iloc[range(0,test_split_sample),[0,1,2,3,4,5,6,7]]
    Y_train = Y.iloc[range(0,test_split_sample),:]

    X_test = X_le.iloc[range(test_split_sample,len(X_le)),[0,1,2,3,4,5,6,7]]
    Y_test = Y.iloc[range(test_split_sample,len(Y)),:]
    
    return X_train, Y_train, X_test, Y_test