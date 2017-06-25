from django.shortcuts import render
import feedparser
import pandas as pd
import numpy as np
import colorsys

def readNews(request):
	d = feedparser.parse('http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml')
	for new in d.entries:
		new.update({'probability' : str(round(np.random.uniform(0, 100), 2)) + '%'})
	df = pd.DataFrame(d.entries)
	filename = 'news.csv'
	df.to_csv(filename, index=False, encoding='utf-8')
	context = {'entries': d.entries}
	return render(request, 'polls/index.html', context)
	
