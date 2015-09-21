#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Kelly Chan
# Date: Oct 4 2014


from django import forms

from hours10k.models import Tracker


class TrackerForm(forms.ModelForm):

    class Meta:
        model = Tracker
        fields = ['goal', 'date', 'title', 'content', 'hours', 'remark']


