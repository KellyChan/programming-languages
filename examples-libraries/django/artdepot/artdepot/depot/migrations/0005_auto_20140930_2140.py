# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('depot', '0004_auto_20140930_2138'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='lineitem',
            name='order',
        ),
        migrations.DeleteModel(
            name='Order',
        ),
        migrations.AlterField(
            model_name='lineitem',
            name='quantity',
            field=models.IntegerField(),
        ),
    ]
