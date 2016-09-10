#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Mood(models.Model):
    user = models.ForeignKey(User)
    date = models.DateField(unique=True)
    rating = models.IntegerField(max_length=1, default=0)
    keywords = models.CharField(max_length=140, null=True, blank=True)
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return r'%d %s %s %d %s %s' \
                % (self.id, self.user, self.date, self.rating, self.remark, self.pubtime)

    class Meta:
        ordering = ['date']


