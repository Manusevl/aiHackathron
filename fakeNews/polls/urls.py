from django.conf.urls import url

from . import rssFeed

urlpatterns = [
    url(r'^$', rssFeed.readNews, name='readNews'),
]