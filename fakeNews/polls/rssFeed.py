from django.shortcuts import render
import feedparser
import pandas as pd
import numpy as np
import colorsys
import string
from urllib.parse import urlparse
import re
from . import AICore

def readNews(request):

	#get news from RSS channel
	d = feedparser.parse('http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml')
	
	#Parse the new one by one
	i = 0
	for i in range(len(d.entries)):
		calcEntries = []
		#adding author to list
		if('author' in d.entries[i]):
			calcEntries.append(d.entries[i].author)
		else:
			calcEntries.append(0)
		# adding language
		a = d.feed.language
		checksum = len(a)
		if(checksum > 2):
			if('language' in d.feed):
				a = d.feed.language.split('-')
				
				calcEntries.append(a[0])
				calcEntries.append(a[1])
			else:
				
				calcEntries.append(0)
				calcEntries.append(0)	
		
		else:
			if('language' in d.feed):
				a = d.feed.language.split('-')
				
				calcEntries.append(a[0])
				calcEntries.append(0)
			else:
				
				calcEntries.append(0)
				calcEntries.append(0)				
		#adding url
		if('link' in d.entries[i]):
			a = urlparse(d.entries[i].link)
			calcEntries.append(a.scheme+'://'+a.netloc)
		else:
			calcEntries.append(0)	
		#adding year,month,day,hour,min
		if('published_parsed' in d.entries[i]):
			a = d.entries[i].published_parsed
			calcEntries.append(d.entries[i].published_parsed.tm_year)
			calcEntries.append(d.entries[i].published_parsed.tm_mon)
			calcEntries.append(d.entries[i].published_parsed.tm_mday)
			calcEntries.append(d.entries[i].published_parsed.tm_hour)
			calcEntries.append(d.entries[i].published_parsed.tm_min)
		else:
			calcEntries.append(0)
			calcEntries.append(0)
			calcEntries.append(0)
			calcEntries.append(0)
			calcEntries.append(0)
		#calculating number of words in text
		if('summary' in d.entries[i]):
			a = d.entries[i].summary 
			calcEntries.append(len(a.split()))	
			calcEntries.append(sum(1 for c in a if c.isupper()))
			calcEntries.append(len(re.findall("!",a)))
		else:
			calcEntries.append(0)
			calcEntries.append(0)
			calcEntries.append(0)
		#calculating number of words,!, and capital letters in title
		if('title' in d.entries[i]):
			a = d.entries[i].title 
			calcEntries.append(len(a.split()))
			calcEntries.append(sum(1 for c in a if c.isupper()))
			calcEntries.append(len(re.findall("!",a)))
		else:
			calcEntries.append(0)
			calcEntries.append(0)
			calcEntries.append(0)
		#Generate CSV File
		df = pd.DataFrame(calcEntries)
		filename = 'oneNew.csv'
		df.to_csv(filename, index=False, encoding='utf-8')
		#Call the model
		
	
	#Get probability
	for new in d.entries:
		new.update({'probability' : str(round(np.random.uniform(0, 100), 2)) + '%'})
	
	
	#Send news to Front-end
	context = {'entries': d.entries}
	return render(request, 'polls/index.html', context)
	
