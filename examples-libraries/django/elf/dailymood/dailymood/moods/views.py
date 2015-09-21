from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.contrib.auth.models import User
from django.http import Http404

from moods.models import Mood
 
 
def IndexView(request):

    moods = Mood.objects.all()

    template_name = 'moods/index.html'
    template_value = {'moods': moods}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

def AddView(request):

    if request.method == "POST":
        thisDate = request.POST['date']
        thisRating = request.POST['rating']
        thisRemark = request.POST['remark']
        thisUser = User.objects.get(id=1)
        mood = Mood(user=thisUser, date=thisDate, rating=thisRating, remark=thisRemark)
        mood.save()

        moods = Mood.objects.all()
        template_name = 'moods/index.html'
        template_value = {'moods': moods}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        template_name = 'moods/add.html'
        template_value = {}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def DeleteView(request, id=''):
    try:
        mood = Mood.objects.get(id=id)
    except Exception:
        raise Http404

    if mood:
        mood.delete()
        return HttpResponseRedirect('/moods')

    moods = Mood.objects.all()
    template_name = 'moods/index.html'
    template_value = {'moods': moods}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def EditView(request, id=''):

    try:
        mood = Mood.objects.get(id=id)
    except Exception:
        raise Http404

    if request.method == "POST":
        mood.rating = request.POST['rating']
        mood.remark = request.POST['remark']
        mood.save()

        moods = Mood.objects.all()
        template_name = 'moods/index.html'
        template_value = {'moods': moods}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

    else:
        template_name = 'moods/edit.html'
        template_value = {'mood': mood}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

