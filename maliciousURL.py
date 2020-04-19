
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
import argparse                                                                 #  utilizing flags to allow user's choice



"""
Choices
    Takes user input as flags and use them in other functions
"""
def choices():
    parser = argparse.ArgumentParser(description='Malicious URL analyzer with Machine Learning')

    parser.add_argument('-t', '--type', required=True, metavar='', help="Input the type of module to use")
    parser.add_argument('-u', '--url', required=True, metavar='', help="Input the URL you want to scan")
    parser.add_argument('-i', '--info', action = infoDisplay(), metavar='', help='Descripton of the Program')

    args = parser.parse_args()

    return args




"""
csvImports Description:
    opens the csv file and store it using pandas.
"""

def csvImport(args):
    # Loading the data utilizing pandas

    print("-----Loading CSV Data-----")
    url_df = pd.read_csv("sample.csv")

    print("\nSample of our data of {}".format(len(url_df)))
    print(url_df.head())

    # Only taking URLs stored to test_url
    """
    if args.url == '':
        test_url = url_df['URLs'][4]
    else:
        test_url = args.url
    """
    if args.url:
        test_url = args.url

    print("\n\n-----UserSelectedURL-----")
    print("-URL: ",test_url)

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
    
    print("\n-----Spliting Train and Testing-----")

    # number set to show before bar graph
    print("\n-Counting splited data frames...\n")
    print("-Training Data Sample: {}".format(len(train_df)))
    print("-Testing Data Sample:  {}".format(len(test_df)))

    return train_df, test_df, labels, test_labels

"""
train_test_graph Description:
    @param: train_df: takes the train_df from train_test method and generate a graph using mathplotlib.py
    @param: test_df: takes the test_df from train_test method and generate a graph using mathplotlib.py
"""
def train_test_graph(train_df, test_df):
    # This is where we generate a bar graph...

    
    print("\n-----Generating testing and training graph-----\n")
    print("---[Exit The Graph to Continue]---")
    # getting count of actual train set and test set
    barGraphTrain = pd.value_counts(train_df['Class'])
    barGraphTest = pd.value_counts(test_df['Class'])

    N = 2
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, barGraphTrain, width, label='Good')

    plt.bar(ind + width, barGraphTest, width, label = 'Bad')
    plt.ylabel('Data sets')
    plt.xlabel('Training/Testing')
    plt.title('Good and Bad URL datasets')
    plt.xticks(ind + width /2, ('Train', 'Test'))
    plt.legend(loc='best')
    #plt.ion()
    #plt.show(block=True)
    plt.show()

    print("-Displayed...")
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

    print("\n-----Vectorizng data frames-----\n")
    print("-May take a minute...\n")

    print("-Training Count Vectorizer...")
    countVecTrain_x = countVec.fit_transform(train_df['URLs'])
    print("Task 1/4 -Complete-\n")

    print("-Training TF-IDF Vectorizer...")
    tfidfVecTrain_x = tfidfVec.fit_transform(train_df['URLs'])
    print("Task 2/4 -Complete-\n")

    print("-Testing Count Vectorizer...")
    countVecTest_x = countVec.transform(test_df['URLs'])
    print("Task 3/4 -Complete-\n")

    print("-Testing TFIDF Vectorizer...")
    tfidfVecTest_x = tfidfVec.transform(test_df['URLs'])
    print("Task 4/4 -Complete-\n")

    print("-Vectorizing complete...\n")

    return countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x, countVec, tfidfVec

def algorithmReport(confuMatrix, score, classReport, ourPrediction):

    confuMatrix = confuMatrix.T

    plt.figure(figsize=(5,5))
    sns.heatmap(confuMatrix, annot=True, fmt="d", lineWidths=.5,square = True, cmap ='Blues', annot_kws={"size": 16}, xticklabels=['bad','good'], yticklabels=['bad', 'good'])
    
    plt.xticks(rotation = 'horizontal', fontsize=16)
    plt.yticks(rotation = 'horizontal', fontsize=16)
    plt.xlabel('Actual label', size = 20)
    plt.ylabel('Predicted Label', size = 20)

    title = 'Accuracy Score:  {0:.4}'.format(score)
    plt.title(title, size = 20)

    print(classReport)
    #plt.ion()
    print("---[Exit the graph to continue]---")

    plt.show()


    print("-Report Generator Completed...\n")
    print("-----Malicious URL Analyzed-----")
    print("-URl Analyzed Result: ", ourPrediction)

def LogicRegTFIDF(labels, test_labels,  tfidfVecTrain_x,  tfidfVecTest_x, test_url, tfidfVec):
    # Training the Logistic Regression Algorithm

    LR_Tfidf = LogisticRegression(solver ='lbfgs')
    LR_Tfidf.fit(tfidfVecTrain_x, labels)

    score_LR_Count = LR_Tfidf.score(tfidfVecTest_x, test_labels)
    predictionsLRTfidf = LR_Tfidf.predict(tfidfVecTest_x)
    cmatrixLRTfidf = confusion_matrix(predictionsLRTfidf, test_labels)
    classReport_LRTfidf = classification_report(predictionsLRTfidf, test_labels)

    test_url = [test_url]
    test_url = tfidfVec.transform(test_url)
    ourPrediction = LR_Tfidf.predict(test_url)

    print("\n-----Model Analyzer-----\n")
    print("-Model generating...\n")
    print("-Logistic Regression w/ TfidfVectorizer")
    algorithmReport(cmatrixLRTfidf, score_LR_Count, classReport_LRTfidf, ourPrediction)
    
# Logistic Regression with Vector Count
def LogRegression_CountVector (labels, test_labels, countVecTrain_x, countVecTest_x, test_url, countVec):
    #train model
    LR_CountVector = LogisticRegression(solver='lbfgs')
    LR_CountVector.fit(countVecTrain_x, labels)

    #test the mode (score, predictions, confusion matrix, classiificattion report)
    score_LR_CountVector = LR_CountVector.score (countVecTest_x, test_labels)
    predictionsCountVector = LR_CountVector.predict(countVecTest_x)
    cmatrixCountVector = confusion_matrix(test_labels, predictionsCountVector)
    creportCountVector = classification_report(test_labels,predictionsCountVector)

    test_url = [test_url]
    test_url = countVec.transform(test_url)
    ourPrediction = LR_CountVector.predict(test_url)
  
    #ourPrediction = ourPrediction.reshape(-1,1)

    print("\n-----Model Analyzer-----\n")
    print("-Model generating...\n")
    print("-Logistic Regression w/ Count Vector")
    algorithmReport(cmatrixCountVector, score_LR_CountVector, creportCountVector, ourPrediction)


def mnbtf(labels, test_labels, tfidfVecTrain_X, tfidfVecTest_x, test_url, tfidfVec):
       # Multinomial Naive Bayesian with TF-IDF
 
   # Train the model
    mnb_tfidf = MultinomialNB()
    mnb_tfidf.fit(tfidfVecTrain_X, labels)
    
    
    # Test the mode (score, predictions, confusion matrix, classification report)
    score_mnb_tfidf = mnb_tfidf.score(tfidfVecTest_x, test_labels)
    predictions_mnb_tfidf = mnb_tfidf.predict(tfidfVecTest_x)
    cmatrix_mnb_tfidf = confusion_matrix(test_labels, predictions_mnb_tfidf)
    creport_mnb_tfidf = classification_report(test_labels, predictions_mnb_tfidf)
    test_url = [test_url]
    test_url = tfidfVec.transform(test_url)

    ourPrediction = mnb_tfidf.predict(test_url)
    
    print("\n-----Model Analyzer-----\n")
    print("-Model generating...\n")
    print("-Multinomial Naive Bayesian w/ TFIDF")
    algorithmReport(cmatrix_mnb_tfidf, score_mnb_tfidf, creport_mnb_tfidf, ourPrediction)
    
def mbbcv(labels, test_labels, countVecTrain_x, countVecTest_x, test_url, countVec):
    # Multinomial Naive Bayesian with Count Vectorizer

    # Train the model
    mnb_count = MultinomialNB()
    mnb_count.fit(countVecTrain_x, labels)


    # Test the mode (score, predictions, confusion matrix, classification report)
    score_mnb_count = mnb_count.score(countVecTest_x, test_labels)
    predictions_mnb_count = mnb_count.predict(countVecTest_x)
    cmatrix_mnb_count = confusion_matrix(test_labels, predictions_mnb_count)
    creport_mnb_count = classification_report(test_labels, predictions_mnb_count)
    test_url = [test_url]
    test_url = countVec.transform(test_url)

    ourPrediction = mnb_count.predict(test_url)

    print("\n-----Model Analyzer-----\n")
    print("-Model generating...\n")
    print("-Multinomial Naive Bayesian w/ Count Vectorizer")
    algorithmReport(cmatrix_mnb_count, score_mnb_count, creport_mnb_count, ourPrediction)

def infoDisplay():
    print("Welcome to malicious URL analyzer 1.0...\n")
    print("Description...\n")
    print("    -t: this flag allows the user to pick a moudle to use to classify good | bad URL\n")
    print("        LGC:  Linear Regression w/ Count Vectorizer\n")
    print("        LGT:  Linear Regression w/ TFIDF Vectorizer\n")
    print("        MNBC: Multinomial Naive Bayesian w/ Count Vectorizer\n")
    print("        MNBT: Multinomial Naive Bayesian w/ TFIDF Vectorizer\n")
    print("    -u: this fflag allows the user to input any url to be analyzed to clasisfy good | bad URL\n")

"""
main Description
    Calls the respective methods and return call by function method
"""
def main():
    args = choices()

    url_df, test_url = csvImport(args)
    
    train_df, test_df, labels, test_labels = train_test(url_df)


    train_test_graph(train_df,test_df)

    countVecTrain_x, tfidfVecTrain_x, countVecTest_x, tfidfVecTest_x, countVec, tfidfVec = vectorizer(train_df, test_df)

    print("-----Initiating Tokenizer-----\n")
    print(test_url)

    print("\n-----Tokenized Result-----\n")
    tokenized_url = tokenizerURL(test_url)
    print(tokenized_url)
    
    print("\n---[Generating Module, please wait]---")
    if(args.info):
        infoDisplay()
    if(args.type == 'LGC'):
        LogRegression_CountVector(labels, test_labels, countVecTrain_x, countVecTest_x, test_url, countVec)
    if(args.type == 'LGT'):
        LogicRegTFIDF(labels, test_labels, tfidfVecTrain_x,  tfidfVecTest_x, test_url, tfidfVec)

    if(args.type == 'MNBC'):
        mbbcv(labels, test_labels, countVecTrain_x,  countVecTest_x, test_url, countVec)

    if(args.type == 'MNBT'):
        mnbtf(labels, test_labels, tfidfVecTrain_x,  tfidfVecTest_x, test_url, tfidfVec)



"""
Calling main
"""
if __name__ == '__main__':
    main()