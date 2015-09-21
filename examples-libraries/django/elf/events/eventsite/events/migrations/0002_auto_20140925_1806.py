# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import datetime


class Migration(migrations.Migration):

    dependencies = [
        ('events', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='event',
            name='pubtime',
            field=models.DateTimeField(default=datetime.date(2014, 9, 25), auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='event',
            name='feedback',
            field=models.TextField(null=True, blank=True),
        ),
    ]
