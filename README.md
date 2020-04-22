Using Machine Learning to Analyze Detection of Malicious URL

Author:

    Andrew Lopez    |       alopez8969@csu.fullerton.edu

    Alex Tran       |       quyen137@csu.fullerton.edu

    Hyun Woo Kim    |       hyunwoo777@csu.fullerton.edu

    Tu Tran         |       trankimtu@csu.fullerton.edu

Summary & Description:

    Implemented 4 data modules: 


        - LGC: Linear Regression w/ Count Vectorizer

        - LGT: Linear Regression w/ TFIDF Vectorizer

        - MNBC: Multinomial Naive Bayesian w/ Count Vectorizer

        - MNBT: Multinomial Naive Bayesian w/ TFIDF Vectorizer


    Description:


        Cybersecurity has been a great issue worldwide. Through this project, we hope to gain

        insights of how important cybersecurity is and get hands on experience into machine 
        
        learning  technologies to also create a URL scanner for malicious urls to prevent 
        
        exploitation and hacking.


    How to compile:

        (Your python 3 interpreter: 3.6 or 3.7 recommended. Not compatible with 3.8)

        python3 ./maliciousURL.py 

            (This allows the user to check options)

        python3 ./maliciousURL.py -t <type> -u <url>
            
            (-t: LGC, LGT, MNBC, MNBT)

            (-u: www.yourwebsite.com/anything-you-want)


Function

        choices: parse the user input as "flags" to connect with other functions

        csvImport: takes in user input url and parse the csv files

        train_test: split the traning and testing into percentile data

        train_test_graph: uses matplotlib.pylot to spit out our dataframe: testing and training of good and bad url

        tokenizerURL: tokenize our input url into meaningful tokens

        vectorizer: vectorize our data from countVectorizer(word count) and  tfidfVectorizer(frequency)

        algorithmReport: generate a report of our data module to Linear Regression and Multinomail Naive Bayes

        LogiRegTFIDF: logistic Regression module with TFIDF vectorizer, attempts to predict our data

        LogRegression_CountVector: logistic Regression module with count vectorizer, attempts to predict our data

        mnbtf: multinomial naive bayes module with TFIDF vectorizer, attempst to predict our data

        mbbcv: multinomial naive bayes module with count vectorizer, attempts to predict our data

        infoDisplay: displays user input choices

        main: passes the correct parameters to the correct function/methods