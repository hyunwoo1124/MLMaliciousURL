import re
def Lexer(url):
    categoryList = re.split(':|\/', url)
    lexeme = [i for i in categoryList if i]
    protocol = lexeme[0]
    domain = lexeme[1]
    sub = ""
    for i in lexeme[2:]:
        sub += i + '/'

    return protocol, domain, sub