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

import seaborn as sns


"""
csvImports Description:
    opens the csv file and store it using pandas.
"""
def csvImport():
    # Loading the data utilizing pandas

    print("#####Loading CSV Data...#####")
    url_df = pd.read_csv("sample.csv")

    # Only taking URLs stored to test_url
    test_url = url_df['URLs'][4]


    print("\nSample of our data of {}".format(len(url_df)))
    print(url_df.head())

    return url_df, test_url
"""
train_test Description:
    @param: url_df: takes the sample.csv and store it in an pandas array
    split the url into training and testing in respective to 80% and 20%
"""
def train_test(url_df):
    # Train = 80% and test = 20%
    test_percentage = .2

    # data must be split between training and testing the sample
    train_df, test_df = train_test_split(url_df, test_size = test_percentage, random_state = 42)

    labels = train_df['Class']
    test_labels = test_df['Class']
    
    print("\n#####Spliting Train and Testing...##### \n")

    # number set to show before bar graph
    print("\nCounting splited data frames...\n")
    print("Training Data Sample: {}".format(len(train_df)))
    print("Testing Data Sample:  {}".format(len(test_df)))

    return train_df, test_df, labels, test_labels

"""
train_test_graph Description:
    @param: train_df: takes the train_df from train_test method and generate a graph using mathplotlib.py
    @param: test_df: takes the test_df from train_test method and generate a graph using mathplotlib.py
"""
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

"""
tokenizerURL Description:
    @param url: takes one of the url from csvImport method from test_url and tokenize the url
"""
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


def vectorizer(train_df,test_df):
     
    countVec = CountVectorizer(tokenizer= tokenizerURL)
    tfidfVec= TfidfVectorizer(tokenizer=tokenizerURL)

    print("\nVectorizng data frames.... may take about a minute...\n")

    print("\nTraining Count Vectorizer...\n")
    countVecTrain_x = countVec.fit_transform(train_df['URLs'])

    print("\nTraining TF-IDF Vectorizer...\n")
    tfidfVecTrain_x = tfidfVec.fit_transform(train_df['URLs'])

    print("\nTesting Count Vectorizer...\n")
    countVecTest_x = countVec.transform(test_df['URLs'])
    print("\nTesting TFIDF Vectorizer...\n")
    tfidfVecTest_x = tfidfVec.transform(test_df['URLs'])

    print("\nVectorizing complete...\n")

    return countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x

def algorithmReport(confuMatrix, score, classReport):
    plt.figure(figsize=(5,5))
    sns.heatmap(confuMatrix, annot=True, fmt="d", lineWidths=.5,square = True, cmap ='Blues', annot_kws={"size": 16}, xticklabels=['bad','good'], yticklabels=['bad', 'good'])
    
    plt.xticks(rotation = 'horizontal', fontsize=16)
    plt.yticks(rotation = 'horizontal', fontsize=16)
    plt.xlabel('Actual label', size = 20)
    plt.ylabel('Predicted Label', size = 20)

    title = 'Accuracy Score:  {0:.4}'.format(score)
    plt.title(title, size = 20)

    print(classReport)
    plt.show()

    print("\nReport Generator Defined...\n")

def LogicRegTFIDF(labels, test_labels, countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x):
    # Training the Logistic Regression Algorithm

    LR_Tfidf = LogisticRegression(solver ='lbfgs')
    LR_Tfidf.fit(tfidfVecTrain_x, labels)

    score_LR_Count = LR_Tfidf.score(tfidfVecTest_x, test_labels)
    predictionsLRTfidf = LR_Tfidf.predict(tfidfVecTest_x)
    cmatrixLRTfidf = confusion_matrix(predictionsLRTfidf, test_labels)
    classReport_LRTfidf = classification_report(predictionsLRTfidf, test_labels)

    print("\nModel generating...\n")
    print("Logistic Regression w/ TfidfVectorizer")
    algorithmReport(cmatrixLRTfidf, score_LR_Count, classReport_LRTfidf)
    

"""
main Description
    Calls the respective methods and return call by function method
"""
def main():
    url_df, test_url = csvImport()
    train_df, test_df, labels, test_labels = train_test(url_df)
    train_test_graph(train_df,test_df)
    
    print("Full URL from the sample...\n")
    print(test_url)

    print("\nURL after tokenizer...\n")
    tokenized_url = tokenizerURL(test_url)
    print(tokenized_url)

    countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x = vectorizer(train_df, test_df)
    LogicRegTFIDF(labels, test_labels, countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x)


"""
Calling main
"""
if __name__ == '__main__':
    main()

# Andrew - MN Baysian
# Alex - 

