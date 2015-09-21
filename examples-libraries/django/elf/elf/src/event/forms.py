from django import forms

from event.models import Event

class EventForm(forms.ModelForm):

    class Meta:
        model = Event
        fields = [
                    'date', 'starttime', 'endtime',
                    'event', 'location', 'organization', 'speakers',
                    'rating', 'feedback'
                 ]

