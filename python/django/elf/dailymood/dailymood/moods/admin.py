#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.contrib import admin

from moods.models import Mood

class MoodAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'rating', 'remark', 'pubtime')
    list_filter = ('pubtime',)
    ordering = ('-pubtime',)


admin.site.register(Mood, MoodAdmin) 
