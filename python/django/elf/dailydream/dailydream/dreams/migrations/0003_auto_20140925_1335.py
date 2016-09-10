# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dreams', '0002_auto_20140925_1329'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dream',
            name='feedback',
            field=models.TextField(null=True, blank=True),
        ),
    ]
