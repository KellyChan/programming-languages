# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dreams', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dream',
            name='feedback',
            field=models.TextField(null=True),
        ),
    ]
