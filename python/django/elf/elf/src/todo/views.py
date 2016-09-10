from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout


from django.http import Http404
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.shortcuts import render, render_to_response


from todo.models import Todo
from todo.forms import TodoForm

#--------------------------------------------------------------#
# URLs
#-------------------------#

URL_INDEX = '/'

def URL_DASHBOARD(username):
    return '/dashboard/%s' % str(username)

def URL_TODO_MANAGE(username):
    return '/dashboard/%s/todo/manage' % str(username)

#--------------------------------------------------------------#
# Todo Manager:
# 
# - (board) ManageView
# - (record) ManageAddView
# - (record) ManageDeleteView, ManageUpdateView
# - (status) ManageDoneView, ManageUndoView
#
#-------------------------#

@login_required
def ManageView(request, username=''):

    form = TodoForm()
    user = User.objects.get(username=username)
    todos = Todo.objects.filter(user_id=user.id, status=1)

    template_name = 'todo/manager.html'
    template_value = {'form': form, 'user': user, 'todos': todos}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

@login_required
def ManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = TodoForm(request.POST or None)
    if form.is_valid():
        thisTodo = form.cleaned_data['todo']
        thisPriority = form.cleaned_data['priority']
        thisUser = User.objects.get(username=username)

        todo = Todo(user=thisUser, todo=thisTodo, priority=thisPriority, status=1)
        todo.save()

        return HttpResponseRedirect(URL_TODO_MANAGE(user.username))

    return HttpResponseRedirect(URL_TODO_MANAGE(user.username))


@login_required
def ManageDeleteView(request, username='', todoid=''):

    user = User.objects.get(username=username)

    try:
        todo = Todo.objects.get(id=todoid)
    except Exception:
        raise Http404

    if todo:
        todo.delete()
    
    return HttpResponseRedirect(URL_TODO_MANAGE(user.username))


@login_required
def ManageUpdateView(request, username='', todoid=''):
 
    todo = Todo.objects.get(id=todoid)
    user = User.objects.get(username=username)

    form = TodoForm(request.POST or None)


    if form.is_valid():
        todo.todo = form.cleaned_data['todo']
        todo.priority = form.cleaned_data['priority']
        todo.save()
        return HttpResponseRedirect(URL_TODO_MANAGE(user.username))
    else:
        template_name = 'todo/manager_update.html'
        template_value = {'form': form, 'todo': todo, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

@login_required
def ManageDoneView(request, username='', todoid=''):

    user = User.objects.get(username=username)
    todo = Todo.objects.get(id=todoid)

    if todo.status == 1:
        todo.status = 0
        todo.save()

    return HttpResponseRedirect(URL_TODO_MANAGE(user.username))


