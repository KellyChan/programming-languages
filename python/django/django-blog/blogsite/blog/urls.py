from django.conf.urls import patterns, url

from blog import views

urlpatterns = patterns('',
    url(r'^$', views.PostView.as_view(), name='index'),
    url(r'^(?P<pk>\d+)$', views.ContentView.as_view(), name='detail'),
    url(r'^archives/$', views.ArchivesView.as_view(), name='archives'),
    url(r'^latest/$', views.LatestView.as_view(), name='latest'),
    #url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
    #url(r'^(?P<pk>\d+)/results/$', views.ResultsView.as_view(), name='results'),
    #url(r'^(?P<question_id>\d+)/vote/$', views.vote, name='vote'),
)
