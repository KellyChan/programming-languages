from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout


from django.http import Http404
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.shortcuts import render, render_to_response


# Goals: 
## Todo
from todo.models import Todo
from todo.forms import TodoForm
## 10K Hours
from hours10k.models import Tracker
from hours10k.forms import TrackerForm
## Event
from event.models import Event
from event.forms import EventForm

# Life:
## Mood
from mood.models import Mood
from mood.forms import MoodForm
## Dream
from dream.models import Dream
from dream.forms import DreamForm
## Diary
from diary.models import Diary
from diary.forms import DiaryForm



#--------------------------------------------------------------#
# URLs
#-------------------------#

URL_INDEX = '/'

def URL_DASHBOARD(username):
    return '/dashboard/%s' % str(username)


#--------------------------------------------------------------#
# Index
#-------------------------#


def IndexView(request):

    template_name = 'master/index.html'
    template_value = locals()
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


#--------------------------------------------------------------#
# Dashboard
#-------------------------#

@login_required
def DashboardView(request, username=''):

    user = User.objects.get(username=username)

    # Goals:
    ## Todo
    todo_form = TodoForm()
    todos = Todo.objects.filter(user_id=user.id, status=1) # list all undos
    ## 10K Hours
    tracker_form = TrackerForm()
    trackers = Tracker.objects.filter(user_id=user.id)
    ## Event
    event_form = EventForm()
    events = Event.objects.filter(user_id=user.id)

    # Life:
    ## Mood
    mood_form = MoodForm()
    moods = Mood.objects.filter(user_id=user.id)
    ## Dream
    dream_form = DreamForm()
    dreams = Dream.objects.filter(user_id=user.id)
    ## Diary
    diary_form = DiaryForm()
    diaries = Diary.objects.filter(user_id=user.id)

    # Rendering template
    template_name = 'master/dashboard.html'
    template_value = {
                        'TodoForm': todo_form, 'todos': todos,
                        'EventForm': event_form, 'events': events,
                        'TrackerForm': tracker_form, 'trackers': trackers, 
                        'MoodForm': mood_form, 'moods': moods,
                        'DreamForm': dream_form, 'dreams': dreams,
                        'DiaryForm': diary_form, 'diaries': diaries,
                     }
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


#--------------------------------------------------------------#
# SignUp, Login, Logout
#-------------------------#

def SignUpView(request):

    if request.user.is_authenticated():
        return HttpResponseRedirect(URL_DASHBOARD(user.username)) 
    else:

        if request.method == 'POST':
            username = request.POST['username']
            password1 = request.POST['password1']
            password2 = request.POST['password2']
            email = request.POST['email']

            user = User.objects.create_user(username=username, email=email, password=password1)

            # authenticate
            new_user = authenticate(username=username, password=password1)
            if new_user is not None:
                login(request, new_user)
                return HttpResponseRedirect(URL_DASHBOARD(user.username))
        else:
            return HttpResponseRedirect(URL_INDEX)


def LoginView(request):

    if request.method == 'POST':

        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user and user.is_active:
            login(request, user)
            return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        return HttpResponseRedirect(URL_INDEX)

def LogoutView(request):
    logout(request)
    return HttpResponseRedirect(URL_INDEX)


#--------------------------------------------------------------#
# Todo Manager:
# 
# - (record) TodoManageAddView
# - (record) TodoManageDeleteView, TodoManageUpdateView
# - (status) TodoManageDoneView, ManageUndoView
#
#-------------------------#


@login_required
def TodoManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = TodoForm(request.POST or None)
    if form.is_valid():
        thisTodo = form.cleaned_data['todo']
        thisPriority = form.cleaned_data['priority']
        thisUser = User.objects.get(username=username)

        todo = Todo(user=thisUser, todo=thisTodo, priority=thisPriority, status=1)
        todo.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def TodoManageDeleteView(request, username='', todoid=''):

    user = User.objects.get(username=username)

    try:
        todo = Todo.objects.get(id=todoid)
    except Exception:
        raise Http404

    if todo:
        todo.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))


@login_required
def TodoManageUpdateView(request, username='', todoid=''):
 
    todo = Todo.objects.get(id=todoid)
    user = User.objects.get(username=username)

    form = TodoForm(request.POST or None)


    if form.is_valid():
        todo.todo = form.cleaned_data['todo']
        todo.priority = form.cleaned_data['priority']
        todo.save()
        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/todo_manage_update.html'
        template_value = {'form': form, 'todo': todo, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

@login_required
def TodoManageDoneView(request, username='', todoid=''):

    user = User.objects.get(username=username)
    todo = Todo.objects.get(id=todoid)

    if todo.status == 1:
        todo.status = 0
        todo.save()

    return HttpResponseRedirect(URL_DASHBOARD(user.username))



#--------------------------------------------------------------#
# 10K Hours Manager:
# 
# - (record) HoursManageAddView
# - (record) HoursManageDeleteView, HoursManageUpdateView
#
#-------------------------#

@login_required
def HoursManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = TrackerForm(request.POST or None)
    if form.is_valid():
        thisGoal = form.cleaned_data['goal']
        thisDate = form.cleaned_data['date']
        thisTitle = form.cleaned_data['title']
        thisContent = form.cleaned_data['content']
        thisHours = form.cleaned_data['hours']
        thisRemark = form.cleaned_data['remark']
        thisUser = User.objects.get(username=username)

        tracker = Tracker(user=thisUser, \
                          goal=thisGoal, \
                          date=thisDate, \
                          title=thisTitle, \
                          content=thisContent, \
                          hours=thisHours, \
                          remark=thisRemark)
        tracker.save()


        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def HoursManageDeleteView(request, username='', trackerid=''):

    user = User.objects.get(username=username)

    try:
        tracker = Tracker.objects.get(id=trackerid)
    except Exception:
        raise Http404

    if tracker:
        tracker.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))

@login_required
def HoursManageUpdateView(request, username='', trackerid=''):
 
    tracker = Tracker.objects.get(id=trackerid)
    user = User.objects.get(username=username)

    form = TrackerForm(request.POST or None)


    if form.is_valid():
        tracker.goal = form.cleaned_data['goal']
        tracker.date = form.cleaned_data['date']
        tracker.title = form.cleaned_data['title']
        tracker.content = form.cleaned_data['content']
        tracker.hours = form.cleaned_data['hours']
        tracker.remark = form.cleaned_data['remark']
        tracker.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/hours10k_manage_update.html'
        template_value = {'form': form, 'tracker': tracker, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))



#--------------------------------------------------------------#
# Mood Manager:
# 
# - (record) MoodManageAddView
# - (record) MoodManageDeleteView, MoodManageUpdateView
#
#-------------------------#


@login_required
def MoodManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = MoodForm(request.POST or None)
    if form.is_valid():
        thisDate = form.cleaned_data['date']
        thisRating = form.cleaned_data['rating']
        thisKeywords = form.cleaned_data['keywords']
        thisUser = User.objects.get(username=username)

        mood = Mood(user=thisUser, date=thisDate, rating=thisRating, keywords=thisKeywords)
        mood.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def MoodManageDeleteView(request, username='', moodid=''):

    user = User.objects.get(username=username)

    try:
        mood = Mood.objects.get(id=moodid)
    except Exception:
        raise Http404

    if mood:
        mood.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))


@login_required
def MoodManageUpdateView(request, username='', moodid=''):
 
    mood = Mood.objects.get(id=moodid)
    user = User.objects.get(username=username)

    form = MoodForm(request.POST or None)


    if form.is_valid():
        mood.date = form.cleaned_data['date']
        mood.rating = form.cleaned_data['rating']
        mood.keywords = form.cleaned_data['keywords']
        mood.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/mood_manage_update.html'
        template_value = {'form': form, 'mood': mood, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

#--------------------------------------------------------------#
# Event Manager:
# 
# - (record) EventManageAddView
# - (record) EventManageDeleteView, EventManageUpdateView
#
#-------------------------#

@login_required
def EventManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = EventForm(request.POST or None)
    if form.is_valid():
        thisDate = form.cleaned_data['date']
        thisStarttime = form.cleaned_data['starttime']
        thisEndtime = form.cleaned_data['endtime']
        thisEvent = form.cleaned_data['event']
        thisLocation = form.cleaned_data['location']
        thisOrganization = form.cleaned_data['organization']
        thisSpeakers = form.cleaned_data['speakers']
        thisRating = form.cleaned_data['rating']
        thisFeedback = form.cleaned_data['feedback']
        thisUser = User.objects.get(username=username)

        event = Event(user=thisUser, 
                      date=thisDate, starttime=thisStarttime, endtime=thisEndtime,
                      event=thisEvent, location=thisLocation, organization=thisOrganization, 
                      speakers=thisSpeakers, rating=thisRating, feedback=thisFeedback)
        event.save()



        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def EventManageDeleteView(request, username='', eventid=''):

    user = User.objects.get(username=username)

    try:
        event = Event.objects.get(id=eventid)
    except Exception:
        raise Http404

    if event:
        event.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))


@login_required
def EventManageUpdateView(request, username='', eventid=''):
 
    event = Event.objects.get(id=eventid)
    user = User.objects.get(username=username)

    form = EventForm(request.POST or None)

    if form.is_valid():
        event.date = form.cleaned_data['date']
        event.starttime = form.cleaned_data['starttime']
        event.endtime = form.cleaned_data['endtime']
        event.event = form.cleaned_data['event']
        event.location = form.cleaned_data['location']
        event.organization = form.cleaned_data['organization']
        event.speakers = form.cleaned_data['speakers']
        event.rating = form.cleaned_data['rating']
        event.feedback = form.cleaned_data['feedback']
        event.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/event_manage_update.html'
        template_value = {'form': form, 'event': event, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))


#--------------------------------------------------------------#
# Dream Manager:
# 
# - (record) DreamManageAddView
# - (record) DreamManageDeleteView, DreamManageUpdateView
#
#-------------------------#

@login_required
def DreamManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = DreamForm(request.POST or None)
    if form.is_valid():
        thisDate = form.cleaned_data['date']
        thisTitle = form.cleaned_data['title']
        thisContent = form.cleaned_data['content']
        thisFeedback = form.cleaned_data['feedback']
        thisUser = User.objects.get(username=username)

        dream = Dream(user=thisUser, date=thisDate, 
                      title=thisTitle, content=thisContent,feedback=thisFeedback)

        dream.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def DreamManageDeleteView(request, username='', dreamid=''):

    user = User.objects.get(username=username)

    try:
        dream = Dream.objects.get(id=dreamid)
    except Exception:
        raise Http404

    if dream:
        dream.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))


@login_required
def DreamManageUpdateView(request, username='', dreamid=''):
 
    dream = Dream.objects.get(id=dreamid)
    user = User.objects.get(username=username)

    form = DreamForm(request.POST or None)

    if form.is_valid():
        dream.date = form.cleaned_data['date']
        dream.title = form.cleaned_data['title']
        dream.content = form.cleaned_data['content']
        dream.feedback = form.cleaned_data['feedback']
        dream.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/dream_manage_update.html'
        template_value = {'form': form, 'dream': dream, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))



#--------------------------------------------------------------#
# Diary Manager:
# 
# - (record) DiaryManageAddView
# - (record) DiaryManageDeleteView, DiaryManageUpdateView
#
#-------------------------#

@login_required
def DiaryManageAddView(request, username=''):

    user = User.objects.get(username=username)
    form = DiaryForm(request.POST or None)
    if form.is_valid():
        thisTitle = form.cleaned_data['title']
        thisContent = form.cleaned_data['content']
        thisUser = User.objects.get(username=username)

        diary = Diary(user=thisUser, title=thisTitle, content=thisContent)
        diary.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))

    return HttpResponseRedirect(URL_DASHBOARD(username))


@login_required
def DiaryManageDeleteView(request, username='', diaryid=''):

    user = User.objects.get(username=username)

    try:
        diary = Diary.objects.get(id=diaryid)
    except Exception:
        raise Http404

    if diary:
        diary.delete()
    
    return HttpResponseRedirect(URL_DASHBOARD(user.username))


@login_required
def DiaryManageUpdateView(request, username='', diaryid=''):
 
    diary = Diary.objects.get(id=diaryid)
    user = User.objects.get(username=username)

    form = DiaryForm(request.POST or None)

    if form.is_valid():
        diary.title = form.cleaned_data['title']
        diary.content = form.cleaned_data['content']
        diary.save()

        return HttpResponseRedirect(URL_DASHBOARD(user.username))
    else:
        template_name = 'master/diary_manage_update.html'
        template_value = {'form': form, 'diary': diary, 'user': user}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))


