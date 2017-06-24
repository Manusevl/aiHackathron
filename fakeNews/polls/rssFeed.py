from django.http import HttpResponse
import feedparser

def readNews(request):
	d = feedparser.parse('http://rss.cnn.com/rss/cnn_latest.rss')
	print(d)
	return HttpResponse("The news has been read")
