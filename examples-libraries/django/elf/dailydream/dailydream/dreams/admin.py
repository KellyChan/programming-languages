from django.contrib import admin

from dreams.models import Dream


class DreamAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'title', 'content', 'feedback', 'pubtime')
    list_filter = ('pubtime',)
    ordering = ('-pubtime',)

admin.site.register(Dream, DreamAdmin)

