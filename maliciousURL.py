#!/usr/bin/python3

# Handles csv files
import pandas as pd
# array
import numpy as np
# Plotting graphs
import matplotlib.pyplot as plt

import random
import re

# sci-kit learning model to train and test split 
from sklearn.model_selection import train_test_split
# sci-kit learning feature with two vectorizer: TfidfVectorizer and CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# importing linear model and naive bayes multinomial
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# helper library to use metric functions
from sklearn.metrics import confusion_matrix, classification_report

#import seaborn as sns

# Loading the data utilizing pandas

print("Loading CSV Data...")
url_df = pd.read_csv("sample.csv")

print("\nSample of our data of {}".format(len(url_df)))
print(url_df.head())

# Train = 80% and test = 20%
test_percentage = .2

# data must be split between training and testing the sample
train_df, test_df = train_test_split(url_df, test_size = test_percentage, random_state = 42)

labels = train_df['Class']
test_tables = train_df['Class']

print("\n Spliting Train and Testing... \n")

# number set to show before bar graph
print("\nCounting splited data frames...\n")
print("Training Data Sample: {}".format(len(train_df)))
print("Testing Data Sample:  {}".format(len(test_df)))

# This is where we generate a bar graph...

barGraphTrain = pd.value_counts(train_df['Class'])
barGraphTrain.plot(kind='bar', fontsize=16)
plt.title("Class Count of Training ", fontsize = 20)
plt.xticks(rotation="horizontal")
plt.xlabel("Class", fontsize=20)
plt.ylabel("Class Count", fontsize=20)

plt.show()

barGraphTest = pd.value_counts(test_df['Class'])
barGraphTest.plot(kind='bar', fontsize=16, colormap='ocean')
plt.title("Class Count of Testing ", fontsize = 20)
plt.xticks(rotation='horizontal')
plt.xticks("Class", fontsize = 20)
plt.yticks("Class Count", fontsiez = 20)

plt.show()