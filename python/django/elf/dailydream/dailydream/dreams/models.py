#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Dream(models.Model):
    user = models.ForeignKey(User)
    date = models.DateField()
    title = models.CharField(max_length=100)
    content = models.TextField()
    feedback = models.TextField(null=True, blank=True)
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'%d %s %s %s' % (self.id, self.title, self.content, self.feedback)


