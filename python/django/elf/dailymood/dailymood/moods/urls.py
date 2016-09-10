#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.conf.urls import patterns, url

from moods import views

urlpatterns = patterns('',
        url(r'^$', views.IndexView, name='index'),
        url(r'^add/$', views.AddView, name='add'),
        url(r'^edit/(?P<id>\d+)/$', views.EditView, name='edit'),
        url(r'^delete/(?P<id>\d+)/$', views.DeleteView, name='delete'),
)
