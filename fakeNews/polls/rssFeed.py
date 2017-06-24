from django.http import HttpResponse
import feedparser
import pandas as pd
import numpy as np

def readNews(request):
	d = feedparser.parse('http://rss.cnn.com/rss/cnn_latest.rss')
	df = pd.DataFrame(d.entries)
	filename = 'news.csv'
	df.to_csv(filename, index=False, encoding='utf-8')
	return HttpResponse("The news has been read")