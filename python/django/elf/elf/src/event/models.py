#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Event(models.Model):

    user = models.ForeignKey(User)
    date = models.DateField()
    starttime = models.TimeField()
    endtime = models.TimeField()
    event = models.CharField(max_length=140)
    location = models.CharField(max_length=140)
    organization = models.CharField(max_length=100)
    speakers = models.CharField(max_length=140)
    rating = models.IntegerField(max_length=1, default=0)
    feedback = models.TextField(null=True, blank=True)
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'%d %s %s %s %s %s %s %s %s %d %d %s %s' \
                % (self.id, self.user, 
                   self.date, self.starttime, self.endtime,
                   self.event, self.location, self.organization, self.speakers, 
                   self.rating, self.feedback, self.pubtime)

    class Meta:
        ordering = ['date', 'starttime', 'pubtime']


