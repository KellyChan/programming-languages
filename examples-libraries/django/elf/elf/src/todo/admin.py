from django.contrib import admin

from todo.models import Todo

class TodoAdmin(admin.ModelAdmin):
    ordering = ('-pubtime',)
    list_filter = ('pubtime',)
    list_display = ('user', 'todo', 'priority', 'status', 'donetime', 'pubtime')

admin.site.register(Todo, TodoAdmin) 
