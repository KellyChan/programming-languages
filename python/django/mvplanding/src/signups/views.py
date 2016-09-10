
from django.conf import settings
from django.contrib import messages
from django.core.mail import send_mail
from django.shortcuts import render, render_to_response, RequestContext, HttpResponseRedirect


from .forms import SignUpForm

def home(request):

    form = SignUpForm(request.POST or None)

    if form.is_valid():
        save_it = form.save(commit=False)
        save_it.save()

        #send_mail(subject, message, from_email, to_list, fail_silently=True)
        subject = 'Thank you for your pre-order from CFE'
        message = 'Welcome to CFE!'
        from_email = settings.EMAIL_HOST_USER
        to_list = [save_it.email, settings.EMAIL_HOST_USER]
        send_mail(subject, message, from_email, to_list, fail_silently=True)

        messages.add_message(request, messages.INFO, 'Thank you for joining.')
        return HttpResponseRedirect('/thank-you')

    template_name = 'signup.html'
    template_value = locals()
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def thankyou(request):

    form = SignUpForm(request.POST or None)

    if form.is_valid():
        save_it = form.save(commit=False)
        save_it.save()
        messages.success(request, 'Thank you for your order, we will be in touch.')
        return HttpResponseRedirect('/thank-you/')

    template_name = 'thankyou.html'
    template_value = locals()
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def aboutus(request):

    template_name = 'aboutus.html'
    template_value = locals()
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

