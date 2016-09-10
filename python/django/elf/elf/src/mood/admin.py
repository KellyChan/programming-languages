#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.contrib import admin

from mood.models import Mood

class MoodAdmin(admin.ModelAdmin):
    ordering = ('-pubtime',)
    list_filter = ('pubtime',)
    list_display = ('user', 'date', 'rating', 'keywords', 'pubtime')


admin.site.register(Mood, MoodAdmin)


