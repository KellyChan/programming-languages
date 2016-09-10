#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.db import models
from django.contrib.auth.models import User  


class Mood(models.Model):
    user = models.ForeignKey(User)
    date = models.DateField()
    rating = models.IntegerField(max_length=1, default=0)
    remark = models.TextField(default='')
    pubtime = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return u'%d %d %s' % (self.id, self.rating, self.remark)


