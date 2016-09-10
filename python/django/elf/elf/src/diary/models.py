#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Diary(models.Model):

    user = models.ForeignKey(User)
    title = models.CharField(max_length=140)
    content = models.TextField()
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'%d %s %s %s %s' \
                % (self.id, self.user, self.title, self.content, self.pubtime)

    class Meta:
        ordering = ['pubtime']


