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
            name='Todo',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('todo', models.CharField(max_length=140)),
                ('priority', models.IntegerField(default=0, max_length=2)),
                ('status', models.IntegerField(default=1, max_length=2)),
                ('donetime', models.DateTimeField()),
                ('pubtime', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['priority', 'pubtime'],
            },
            bases=(models.Model,),
        ),
    ]
