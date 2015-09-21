#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Kelly Chan
# Date: Oct 4 2014

from django.db import models
from django.contrib.auth.models import User  


class Tracker(models.Model):

    user = models.ForeignKey(User)
    goal = models.CharField(max_length=100)
    date = models.DateField()
    title = models.CharField(max_length=140)
    content = models.TextField()
    hours = models.DecimalField(max_digits=5, decimal_places=2)
    remark = models.TextField(null=True, blank=True)
    pubtime = models.DateTimeField(auto_now_add=True)
    
    def __unicode__(self):
        return u'%d %s %s %s %s %s %f %s %s' \
                % (self.id, self.user, self.goal, self.date, self.title, self.content, self.hours, self.remark, self.pubtime)

    class Meta:
        ordering = ['goal', 'pubtime']


