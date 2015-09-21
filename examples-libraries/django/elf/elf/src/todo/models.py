#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Todo(models.Model):

    user = models.ForeignKey(User)
    todo = models.CharField(max_length=140)
    priority = models.IntegerField(max_length=2, default=0)
    status = models.IntegerField(max_length=2, default=1) # 1 for undo, 0 for done
    donetime = models.DateTimeField(auto_now_add=True)
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'%d %s %s %d %d %s %s' \
                % (self.id, self.user, self.todo, self.priority, self.status, self.donetime, self.pubtime)

    class Meta:
        ordering = ['priority', 'pubtime']

