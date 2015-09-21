from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.contrib.auth.models import User
from django.http import Http404

from todo.models import Todo
 
 
def TodoView(request):
    template_name = 'todo/todo.html'

    todolist = Todo.objects.filter(flag=1)
    finishedlist = Todo.objects.filter(flag=0)
    template_value = {'todolist': todolist, 'finishedlist': finishedlist}

    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

def AddView(request):

    if request.method == "POST":
        thisTodo = request.POST['todo']
        thisPriority = request.POST['priority']
        thisUser = User.objects.get(id=1)
        todo = Todo(user=thisUser, todo=thisTodo, priority=thisPriority, flag=1)
        todo.save()

        todolist = Todo.objects.filter(flag=1)
        finishedlist = Todo.objects.filter(flag=0)
        template_name = 'todo/todo.html'
        template_value = {'todolist': todolist, 'finishedlist': finishedlist}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        todo = ''
        template_name = 'todo/add.html'
        template_value = {'todo': todo}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
        

def DoneView(request, id=''):

    todo = Todo.objects.get(id=id)
    if todo.flag == 1:
        todo.flag = 0
        todo.save()
        return HttpResponseRedirect('/todo')

    todolist = Todo.objects.filter(flag=1)
    finishedlist = Todo.objects.filter(flag=0)
    template_name = "todo/todo.html"
    template_value = {'todolist': todolist, 'finishedlist': finishedlist}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def UndoView(request, id=''):

    todo = Todo.objects.get(id=id)
    if todo.flag == 0:
        todo.flag = 1
        todo.save()
        return HttpResponseRedirect('/todo')

    todolist = Todo.objects.filter(flag=1)
    finishedlist = Todo.objects.filter(flag=0)
    template_name = "todo/todo.html"
    template_value = {'todolist': todolist, 'finishedlist': finishedlist}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

def DeleteView(request, id=''):
    try:
        todo = Todo.objects.get(id=id)
    except Exception:
        raise Http404

    if todo:
        todo.delete()
        return HttpResponseRedirect('/todo')

    todolist = Todo.objects.filter(flag=1)
    template_name = "todo/todo.html"
    template_value = {'todolist': todolist}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

def EditView(request, id=''):

    try:
        todo = Todo.objects.get(id=id)
    except Exception:
        raise Http404

    if request.method == "POST":
        todo.todo = request.POST['todo']
        todo.priority = request.POST['priority']
        todo.save()

        todolist = Todo.objects.filter(flag=1)
        finishedlist = Todo.objects.filter(flag=0)
        template_name = 'todo/todo.html'
        template_value = {'todolist': todolist, 'finishedlist': finishedlist}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        template_name = 'todo/edit.html'
        template_value = {'todo': todo}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

 
