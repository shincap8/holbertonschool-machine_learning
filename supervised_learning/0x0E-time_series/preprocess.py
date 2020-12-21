#!/usr/bin/env python3
"""Method to preprocess the data"""
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd


def preprocess():
    """Method to preprocess the data"""
    filename = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    df = pd.read_csv(filename)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    df['Weighted_Price'].fillna(value=0, inplace=True)
    df['Open'].fillna(method='ffill', inplace=True)
    df['High'].fillna(method='ffill', inplace=True)
    df['Low'].fillna(method='ffill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)
    df = df[0::60]
    datetime = pd.to_datetime(df.pop('Timestamp'))
    variables = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
                 'Volume_(Currency)', 'Weighted_Price']
    plot_features = df[variables]
    plot_features.index = datetime
    _ = plot_features.plot(subplots=True)
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    plot_features = train_df[variables]
    plot_features.index = datetime[0:int(n * 0.7)]
    _ = plot_features.plot(subplots=True)
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return (train_df, val_df, test_df)
