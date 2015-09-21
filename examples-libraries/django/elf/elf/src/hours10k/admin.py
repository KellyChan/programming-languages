#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Kelly Chan
# Date: Oct 4 2014

from django.contrib import admin

from hours10k.models import Tracker

class TrackerAdmin(admin.ModelAdmin):
    ordering = ('-pubtime',)
    list_filter = ('pubtime',)
    list_display = ('user', 'goal', 'date', 'title', 'content', 'hours', 'remark', 'pubtime')

admin.site.register(Tracker, TrackerAdmin) 
