import argparse
import requests
import validators
import yaml
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment

"""
This will load up the url fata from the csv file and print it out
"""
data_dir = "URLData.csv"
print("- Loading the CSV Data -")
url_df = pd.read_csv(data_dir)

print("\n### CSV Data is Loaded ###\n")

print(url_df)
