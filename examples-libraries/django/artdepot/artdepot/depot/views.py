import datetime
 
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required  
from django.contrib.auth.models import User

from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.http import Http404

from depot.models import *


#--------------------------------------------------------------------#
# Store Views
#--------------#

def StoreDashboardView(request):

    if request.method == 'POST':

        thisTitle = request.POST['title']
        thisDescription = request.POST['description']
        thisPrice = request.POST['price']
        thisImageURL = request.POST['image_url']
        thisTimeAvailabled = request.POST['time_availabled']
        thisTimePublished = datetime.datetime.now()

        product = Product(title=thisTitle, \
                          description=thisDescription, \
                          price=thisPrice, \
                          image_url=thisImageURL, \
                          time_availabled=thisTimeAvailabled, \
                          time_published=thisTimePublished)
        product.save()
        products = Product.objects.all()

        template_name = 'depot/store_dashboard.html'
        template_value = {'products': products}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))
    
    else:
        products = Product.objects.all()

        template_name = 'depot/store_dashboard.html'
        template_value = {'products': products}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))



def ProductUpdateView(request, id=''):

    try: 
        product = Product.objects.get(id=id)
    except:
        raise Http404

    if request.method == 'POST':

        product.title = request.POST['title']
        product.description = request.POST['description']
        product.price = request.POST['price']
        product.image_url = request.POST['image_url']
        product.time_availabled = request.POST['time_availabled']

        product.save()
        return HttpResponseRedirect('/depot/store/dashboard')

    else:
        template_name = 'depot/product_update.html'
        template_value = {'product': product}
        return render_to_response(template_name, template_value, context_instance=RequestContext(request))

 
def ProductDeleteView(request, id=''):

    try:
        product = Product.objects.get(id=id)
    except Exception:
        raise Http404

    if product:
        product.delete()
    
    return HttpResponseRedirect('/depot/store/dashboard')


#--------------------------------------------------------------------#
# User Views
#--------------#

def ProductView(request):

    products = Product.objects.all()

    template_name = 'depot/products.html'
    template_value = {'products': products}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))

@login_required
def UserDashboardView(request, id=''):

    user = User.objects.get(id=id)
    cart = request.session.get('cart', None)

    if not cart:
        cart = Cart()    
        request.session['cart'] = cart

    template_name = 'depot/user_dashboard.html'
    template_value = {'cart': cart}
    return render_to_response(template_name, template_value, context_instance=RequestContext(request))


def CartAddView(request, id=''):

    product = Product.objects.get(id=id)
    cart = request.session.get('cart', None)

    if not cart:
        cart = Cart()
        request.session['cart'] = cart

    cart.add_product(product)
    request.session['cart'] = cart  # update cart
    return HttpResponseRedirect('/depot')

def CartRemoveView(request, id=''):
    pass

def CartCleanView(request):

    request.session['cart'] = Cart()
    return HttpResponseRedirect('/depot/user/dashboard')

def OrderCreateView(request):

    if request.method == 'POST':

        thisName = request.POST['name']
        thisAddress = request.POST['address']
        thisEmail = request.POST['email']

        order = Order(name=thisName, address=thisAddress, email=thisEmail)
        order.save()

        return HttpResponseRedirect('/depot/user/dashboard')

    else:
        return HttpResponseRedirect('/depot/user/dashboard')

#--------------------------------------------------------------------#
# Account Login/Logout
#--------------#

def SigninView(request):

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user and user.is_active:
            login(request, user)
            return HttpResponseRedirect('/depot/user/dashboard/user=%s/' % user.id)
            #return UserDashboardView(request, id=user.id)

    else:
        return HttpResponseRedirect('/depot')


def SignoutView(request):
    logout(request)
    return HttpResponseRedirect('/depot')


def SignupView(request):

    if request.user.is_authenticated():
        return HttpResponseRedirect('/depot/user/dashboard')
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
                return HttpResponseRedirect('/depot/user/dashboard')
                  
        else:
            template_name = 'signup.html'
            template_value = {}
            return render_to_response(template_name, template_value, context_instance=RequestContext(request))
