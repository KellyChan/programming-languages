# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('depot', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='LineItem',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('unit_price', models.DecimalField(max_digits=8, decimal_places=2)),
                ('quantity', models.IntegerField()),
                ('product', models.ForeignKey(to='depot.Product')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
