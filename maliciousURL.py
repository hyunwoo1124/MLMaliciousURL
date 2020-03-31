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

# getting count of actual train set and test set
barGraphTrain = pd.value_counts(train_df['Class'])
barGraphTest = pd.value_counts(test_df['Class'])

N = 2
ind = np.arange(N)
width = 0.35

plt.bar(ind, barGraphTrain, width, label='Train')
plt.bar(ind + width, barGraphTest, width, label = 'Test')

plt.ylabel('Data sets')
plt.xlabel('Training/Testing')
plt.title('Good and Bad URL datasets')
plt.xticks(ind + width /2, ('Train', 'Test'))
plt.legend(loc='best')
plt.show()

