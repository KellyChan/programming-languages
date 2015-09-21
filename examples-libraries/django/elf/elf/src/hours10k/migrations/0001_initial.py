# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Tracker',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('goal', models.CharField(max_length=100)),
                ('date', models.DateField()),
                ('title', models.CharField(max_length=140)),
                ('content', models.TextField()),
                ('hours', models.DecimalField(max_digits=5, decimal_places=2)),
                ('remark', models.TextField(null=True, blank=True)),
                ('pubtime', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['goal', 'pubtime'],
            },
            bases=(models.Model,),
        ),
    ]
