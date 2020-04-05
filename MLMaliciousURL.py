import argparse
import requests
import validators
import yaml
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment

"""
The following code is basically a clean version of the parsing cmd line interface
For example -h will bring help page, -v will bring version... This is what im doing
underneath
"""


parser = argparse.ArgumentParser(description='Machine Learning Malicious URL Detector Version 1.0')
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-url', type=str, help='Type the URL to be analyzed')
parser.add_argument('--config', help = 'Path to configuration file')
parser.add_argument('-o', '--output', help='Allows you to write to a file')

args = parser.parse_args()

"""
As we add more features the config file dictionary will be larger and larger
This is what we have so far
1. Form:        Form ensures we are using the proper protocol such as https not HTTP
2. Comment
3. passwords
"""

config = {'forms': True, 'comments': True, 'passwords': True}   # if more functions we can add them here

if(args.config):
    print('Using config file: ' + args.config)
    config_file = open(args.config, 'r')
    config_from_file = yaml.load(config_file)
    if(config_from_file):
        config = {**config, **config_from_file}

report = ''

url = args.url

if(validators.url(url)):
    #Requesting url in html document
    result_html = requests.get(url).text
    #HTML is in garbage syntax so BeautifulSoup helps you organize them
    parsed_html = BeautifulSoup(result_html, 'html.parser')

    #looking for keyword form
    forms = parsed_html.find_all('form')
    #searching for  all comments
    comments = parsed_html.find_all(string=lambda text: isinstance(text,Comment))
    #Searching for input in javascript that has name and password
    password_inputs = parsed_html.find_all('input', {'name': 'password'})

    """
    This is where we grab the config dictionaries. For example since by default all config
    are True, it will do the following code. 
    """

    #If the header is not https then its not secure
    if(config['forms']):
        for form in forms:
            if(form.get('action').find('https') < 8) and (urlparse(url).scheme != 'https'):
                report += 'Form Issue: Insecure URL, not using security protocol ' + form.get('action') + ' found in document\n'

    # Scan through all the HTML document and if perhaps developer or the owner of the webpage has hidden comment section
    # That may have a key it will detect it by this code underneath
    if(config['comments']):
        for comment in comments:
            if(comment.find('key: ' ) > -1):
                report += 'Comment Issue: Insecure form action ' + form.get('action') + ' found in document\n'

    if(config['passwords']):
        for password_input in password_inputs:
            if(password_input.get('type') != 'passwords'):
                report += 'Input Issue: Plaintext password input found. Can be sniffed.'
else:
    print('Invalid URL. Please include full URL including scheme')

if(report ==''):
    report += 'No Malicious Vulnerability Found'
else:
    header = 'Machine Learning URL Detector Analysis:\n'
    header += '====================================================\n'

    report = header + report

print("HTML contents......................................\n")
print("=======================================================\n")
print(requests.get(url).text)
print("=======================================================\n")
print("HTML contents end of file\n")

print(report)


if(args.output):
    f = open(args.output, 'w')
    f.write(requests.get(url).text)
    f.write(report)
    f.close
    print('Report saved to: ' + args.output)