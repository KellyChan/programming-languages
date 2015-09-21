#!/usr/bin/python
# -*- coding: utf-8 -*-

from django.conf.urls import patterns, url

from depot import views

urlpatterns = patterns('',

        # Store View
        #url(r'^store/$', views.StoreView, name='store'),
        url(r'^store/dashboard/$', views.StoreDashboardView, name='store-dashboard'),
        url(r'^store/dashboard/products/(?P<id>\d+)/update/$', views.ProductUpdateView, name='product-update'),
        url(r'^store/dashboard/products/(?P<id>\d+)/delete/$', views.ProductDeleteView, name='product-delete'),


        # Front End
        url(r'^$', views.ProductView, name='index'),


        # Account
        url(r'^account/signup$', views.SignupView, name='signup'),
        url(r'^account/signin$', views.SigninView, name='signin'),
        url(r'^account/signout$', views.SignoutView, name='signout'),
        
        # User
        url(r'^user/dashboard/user=(?P<id>\d+)/$', views.UserDashboardView, name='user-dashboard'),
        url(r'^user/dashboard/cart/add/product=(?P<id>\d+)$', views.CartAddView, name='cart-add'),
        #url(r'^user/dashboard/cart/remove/product=(?P<id>\d+)$', views.CartRemoveView, name='cart-remove'),
        url(r'^user/dashboard/cart/clean$', views.CartCleanView, name='cart-clean'),
        url(r'^user/dashboard/order/create$', views.OrderCreateView, name='order-create'),
)

