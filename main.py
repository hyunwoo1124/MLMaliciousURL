import numpy as np
import pandas as pd
from lexer import Lexer
# import lexer
# Read in web library file

def header(msg):
    print('-' * 50)
    print(f'[ {msg} ]')
    print('-' * 50)

def inputLib():
    filename = 'weblib.txt'
    df = pd.read_csv(filename)

    # header("4. df.dtypes")
    # print(df.dtypes)

    # header("4. df.index")
    # print(df.index)

    # header("4. df.columns")
    # print(df.columns)

    # header("4. df.values")
    # print(df.values)

    # dfValue = df.values
    # print ('dfValue = ', dfValue)
    # for i in dfValue:
    #     print(dfValue[0])
    # print(df.loc['domain' == 'www.badweb.com'])
    # print(df[df['domain'].isin(['www.badweb.com'])])
    # print (df.domain == 'www.google.com')
    # print (df.domain == 'www.googles.com')
        
    return df

def checkWebsite(myProtocol, myDomain):
    df = inputLib()
    # print(df.protocol)
    # print(df.domain)
    isInLib = (df.protocol == myProtocol) & (df.domain == myDomain)
    result = df[isInLib]
    status = result.status
    for i in status:
        if i == True:
            print (f'The website {myDomain} is safe to browse')
        else:
            print (f'The website {myDomain} is NOT safe to browse')

    # print(status)
    # print(result['status'])
    # for i in isInLib:
        # print (i)
    # isInLib = (df.protocol == myProtocol) 
    
    # result = df[isInLib]
    # print(isInLib)
    # status = result['status']
    # print(result)
    # print(status[values])
    # return True

def main():
    # weblib = inputLib()
    url = input('Enter URL: ')
    protocol, domain, sub = Lexer(url)
    checkWebsite(protocol, domain)
    # checkWebsite('https', 'www.badweb.com')

    # print(lexemes)
    # protocol = lexemes[0]
    # domain = lexemes[1]
    # sub = ""
    # i = 2
    # while i < lexemes.len():
    #     sub += '/' + lexemes[i]
    # print('sub = ', sub)


    # checkWebsite('http', 'www.google.com')
    
    # print(a)
    # print(weblib)


main() 


