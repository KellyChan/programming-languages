from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.contrib.auth.models import User
from django.http import Http404

from dreams.models import Dream


def IndexView(request):

    dreams = Dream.objects.all()

    template_name = 'dreams/index.html'
    template_value = {'dreams': dreams}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def EditView(request, id=''):

    try:
        dream = Dream.objects.get(id=id)
    except Exception:
        raise Http404

    if request.method == "POST":
        dream.title = request.POST['title']
        dream.content = request.POST['content']
        dream.feedback = request.POST['feedback']
        dream.save()

        dreams = Dream.objects.all()
        template_name = 'dreams/index.html'
        template_value = {'dreams': dreams}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        template_name = 'dreams/edit.html'
        template_value = {'dream': dream}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def DeleteView(request, id=''):
    try:
        dream = Dream.objects.get(id=id)
    except Exception:
        raise Http404

    if dream:
        dream.delete()
        return HttpResponseRedirect('/dreams')

    dreams = Dream.objects.all()
    template_name = 'dreams/index.html'
    template_value = {'dreams': dreams}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def AddView(request):

    if request.method == "POST":
        thisDate = request.POST['date']
        thisTitle = request.POST['title']
        thisContent = request.POST['content']
        thisFeedback = request.POST['feedback']
        thisUser = User.objects.get(id=1)
        dream = Dream(user=thisUser, date=thisDate, title=thisTitle, content=thisContent, feedback=thisFeedback)
        dream.save()

        dreams = Dream.objects.all()
        template_name = 'dreams/index.html'
        template_value = {'dreams': dreams}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        template_name = 'dreams/add.html'
        template_value = {}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
