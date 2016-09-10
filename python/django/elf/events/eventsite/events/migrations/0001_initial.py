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
            name='Event',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('date', models.DateField()),
                ('starttime', models.TimeField()),
                ('endtime', models.TimeField()),
                ('event', models.CharField(max_length=140)),
                ('location', models.CharField(max_length=140)),
                ('organization', models.CharField(max_length=100)),
                ('speakers', models.CharField(max_length=140)),
                ('status', models.IntegerField(default=0, max_length=1)),
                ('rating', models.IntegerField(default=0, max_length=1)),
                ('feedback', models.TextField()),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
