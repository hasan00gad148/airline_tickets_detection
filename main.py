import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def stopcorrect(x):
    e = x[0:1]
    if e == 'n':
        e = 0
    c = int(e)
    return c


def dS(y):
    f = y[12:len(y) - 1]
    return f


def dS2(y):
    f = y[17:len(y) - 2]
    return f

def preproc( dataset2):

    dataset = pd.read_csv(dataset2)

    # handle date column
    dataset["date"] = dataset["date"].apply(str)
    dataset["date"] = dataset["date"].str.replace("/", "-")
    dataset[["day", "month", "year"]] = dataset['date'].str.split("-", expand=True)
    dataset.drop(["date", "year"], axis=1, inplace=True)

    # handle time taken column

    # 1- split (06h 30m) => (06h)(30m) => (06)(h)(30)(m)
    dataset[["time_taken_h", "time_taken_m"]
    ] = dataset['time_taken'].str.split(" ", expand=True)
    dataset[["time_taken_h_num", "time_taken_h_label"]
    ] = dataset['time_taken_h'].str.split("h", expand=True)
    dataset[["time_taken_m_num", "time_taken_m_label"]
    ] = dataset['time_taken_m'].str.split("m", expand=True)

    # 2- convert hours to minutes & sum it with minute column => result = (06*60)+30 = 390 minute
    dataset["time_taken_m_num"] = dataset["time_taken_m_num"].replace('', '0')
    dataset["time_taken_minutes"] = dataset["time_taken_h_num"].astype(
        float) * 60 + dataset["time_taken_m_num"].astype(float)

    # 3- Drop columns
    dataset.drop(["time_taken", "time_taken_h_label", "time_taken_h", "time_taken_m",
                  "time_taken_m_label", "time_taken_h_num", "time_taken_m_num"], axis=1, inplace=True)

    # handel dep_time column
    dataset["dep_hour"] = pd.to_datetime(dataset["dep_time"]).dt.hour
    dataset["dep_min"] = pd.to_datetime(dataset["dep_time"]).dt.minute

    # handel arr_time column
    dataset["arr_hour"] = pd.to_datetime(dataset["arr_time"]).dt.hour
    dataset["arr_min"] = pd.to_datetime(dataset["arr_time"]).dt.minute

    # Drop columns
    dataset.drop(["dep_time", "arr_time"], axis=1, inplace=True)

    # modify in coloumn 'stop' {remove spaces }
    dataset['stop'] = dataset['stop'].str.strip()

    # modify in coloumn 'stop' {remove some words }
    dataset['stop'] = dataset['stop'].apply(stopcorrect)

    # split coloumn 'route' into two coloumns by ','
    dataset[['source', 'destination']] = dataset.route.str.split(",", expand=True)

    # remove some words
    dataset['source'] = dataset['source'].apply(dS)
    dataset['destination'] = dataset['destination'].apply(dS2)

    # drop coloumn 'route'
    dataset.drop(['route'], axis=1,inplace=True)

    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'source'.
    dataset['source'] = label_encoder.fit_transform(dataset['source'])
    dataset['source'].unique()

    # Encode labels in column 'destination'.
    dataset['destination'] = label_encoder.fit_transform(dataset['destination'])
    dataset['destination'].unique()

    # Encode labels in column 'type'.
    dataset['type'] = label_encoder.fit_transform(dataset['type'])
    dataset['type'].unique()

    # Encode labels in column 'ch_code'.
    dataset['ch_code'] = label_encoder.fit_transform(dataset['ch_code'])
    dataset['ch_code'].unique()

    # Encode labels in column 'airline'.
    dataset['airline'] = label_encoder.fit_transform(dataset['airline'])
    dataset['airline'].unique()

    # Encode labels in column 'TicketCategory'.
    dataset['TicketCategory'] = label_encoder.fit_transform(dataset['TicketCategory'])
    dataset['TicketCategory'].unique()



    return dataset

#data2 = preproc("airline-price-classification.csv")
