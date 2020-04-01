#!/usr/bin/python3

import pandas as pd                                                             # Handles csv files
import numpy as np                                                              # array


import matplotlib.pyplot as plt                                                 # Graph generating library

import random                                                                   # importing random and regEx
import re

from sklearn.model_selection import train_test_split                            # sci-kit learning model to train and test split 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer    # sci-kit learning feature with two vectorizer: TfidfVectorizer and CountVectorizer
from sklearn.linear_model import LogisticRegression                             # importing linear model and naive bayes multinomial
from sklearn.naive_bayes import MultinomialNB

# helper library to use metric functions
from sklearn.metrics import confusion_matrix, classification_report             # helper library to use metric functions

#import seaborn as sns


def csvImport():
    # Loading the data utilizing pandas

    print("#####Loading CSV Data...#####")
    url_df = pd.read_csv("sample.csv")

    # Only taking URLs stored to test_url
    test_url = url_df['URLs'][4]


    print("\nSample of our data of {}".format(len(url_df)))
    print(url_df.head())

    return url_df, test_url

def train_test(url_df):
    # Train = 80% and test = 20%
    test_percentage = .2

    # data must be split between training and testing the sample
    train_df, test_df = train_test_split(url_df, test_size = test_percentage, random_state = 42)

    #labels = train_df['Class']
    #test_labels = test_df['Class']
    
    print("\n#####Spliting Train and Testing...##### \n")
    print("#                                          #")
    print("############################################")

    # number set to show before bar graph
    print("\nCounting splited data frames...\n")
    print("Training Data Sample: {}".format(len(train_df)))
    print("Testing Data Sample:  {}".format(len(test_df)))

    return train_df, test_df


def train_test_graph(train_df, test_df):
    # This is where we generate a bar graph...

    print("\n#####Generating testing and training graph#####\n")
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


# Implementing tokenizer for the URL
def tokenizerURL(url):
    """
    This method will split the url into tokenized forms:
    www.example.com/this
        ex: example,this
    (tokenizer will remove www, . , -, /)
    """
    #Utilizing regEX to get [-,/] from the url
    tokens = re.split('[/-]', url)

    # for loop to iterate the whole url section
    for tok in tokens:
        if tok.find(".") >= 0:
            # splits the subdomains
            dotSplit = tok.split('.')
            # Remove top level domain .com, .edu, .gov
            # and www. since they're too common
            if "com" in dotSplit:
                dotSplit.remove("com")
            if "edu" in dotSplit:
                dotSplit.remove("edu")
            if "gov" in dotSplit:
                dotSplit.remove("gov")
            if "www" in dotSplit:
                dotSplit.remove("www")
            if "org" in dotSplit:
                dotSplit.remove("org")
            
            tokens += dotSplit

        return tokens

    print ("Tokenizer in transit...\n")


def main():
    url_df, test_url = csvImport()
    train_df, test_df = train_test(url_df)
    train_test_graph(train_df,test_df)
    
    print("Full URL from the sample...\n")
    print(test_url)

    print("\nURL after tokenizer...\n")
    tokenized_url = tokenizerURL(test_url)
    print(tokenized_url)


if __name__ == '__main__':
    main()




