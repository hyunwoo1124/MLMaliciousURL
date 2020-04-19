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
