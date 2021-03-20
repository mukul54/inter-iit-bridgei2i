import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Reading csv file which has websites to scrape and score(to judge the article)
# Don't forgot to edit the file location as per your needs
data=['https://hindi.gadgets360.com/mobiles/news']

	# CAUTION: There might be some redirecting websites which mght cause a problem and 
	# some site might block because requesting to many times.
	# "SO PLEASE KEEP AN EYE WHILE SCRAPING"

#data["text"] = ""
#  scraping
for i in range(1):
    print(i)
    try:
        result=requests.get(str(data[i]))
    except Exception:
        print(str(i)+" - error")
        continue
    src=result.content
    soup=BeautifulSoup(src,'lxml')   
    head=[]
    head = [i['href'] for i in soup.find_all('a', href=True)]

h = []

for i in head:
    if i.startswith("https://hindi.gadgets360.com/mobiles") is True:
        h.append(i)

s = "https://hindi.gadgets360.com/mobiles/news/page-"
for i in range(2,30):
    data = s+str(i)
    try:
        result=requests.get(str(data))
    except Exception:
        print(str(i)+" - error")
        continue
    src=result.content
    soup=BeautifulSoup(src,'lxml')   
    head=[]
    head = [k['href'] for k in soup.find_all('a', href=True)]
    for j in head:
        if j.startswith("https://hindi.gadgets360.com/mobiles") is True:
            h.append(j)

df = pd.DataFrame(h, columns=["colummn"])
df.to_csv('list1.csv', index=False)

    
# PREPROCESSING

