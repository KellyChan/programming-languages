from django import forms

from dream.models import Dream

class DreamForm(forms.ModelForm):

    class Meta:
        model = Dream
        fields = ['date', 'title', 'content', 'feedback']


