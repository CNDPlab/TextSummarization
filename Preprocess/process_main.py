from configs import Config
from bs4 import BeautifulSoup

args = Config()

with open(args.raw_file) as f:
    data = f.read()

soup = BeautifulSoup(data,'lxml')
