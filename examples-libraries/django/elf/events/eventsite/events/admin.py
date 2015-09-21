#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.contrib import admin

from events.models import Event


class EventAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'starttime', 'endtime', 
                    'event', 'location', 'organization', 'speakers', 
                    'status', 'rating', 'feedback', 'pubtime')

    list_filter = ('pubtime',)
    ordering = ('-pubtime',)


admin.site.register(Event, EventAdmin)

