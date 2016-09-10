from django.utils import timezone
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.views import generic

from blog.models import Post


class PostView(generic.ListView):
    template_name = 'blog/posts.html'
    context_object_name = 'object_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return Post.objects.all().order_by('-date')[:5]

class ContentView(generic.DetailView):

    template_name = 'blog/detail.html'
    model = Post

class ArchivesView(generic.ListView):

    template_name = 'blog/archives.html'
    model = Post

    def get_queryset(self):
        return Post.objects.all().order_by("-date")


class LatestView(generic.ListView):

    template_name = 'blog/latest.html'
    context_object_name = 'object_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return Post.objects.all().order_by('-date')[:5]

