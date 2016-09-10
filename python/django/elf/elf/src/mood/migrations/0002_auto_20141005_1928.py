# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mood', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mood',
            name='remark',
        ),
        migrations.AddField(
            model_name='mood',
            name='keywords',
            field=models.CharField(max_length=140, null=True, blank=True),
            preserve_default=True,
        ),
    ]
