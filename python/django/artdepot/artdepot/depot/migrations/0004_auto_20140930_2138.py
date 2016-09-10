# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('depot', '0003_auto_20140930_2137'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lineitem',
            name='quantity',
            field=models.IntegerField(default=1),
        ),
    ]
