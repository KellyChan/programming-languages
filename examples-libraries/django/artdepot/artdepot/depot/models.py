from django.contrib.auth.models import User  
from django.db import models

class Product(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    image_url = models.CharField(max_length=200)
    time_availabled = models.DateTimeField()
    time_published = models.DateTimeField(auto_now_add=True)



class Order(models.Model):
    name = models.CharField(max_length=50)
    address = models.TextField()
    email = models.EmailField()



class LineItem(models.Model):
    product = models.ForeignKey(Product)
    order = models.ForeignKey(Order)
    unit_price = models.DecimalField(max_digits=8, decimal_places=2)
    quantity = models.IntegerField()


class Cart(object):

    def __init__(self, *args, **kwargs):
        self.items = []
        self.total_price = 0

    def add_product(self, product):
        self.total_price += product.price
        self.items.append(LineItem(product=product, unit_price=product.price, quantity=1))


class UserProfile(models.Model):
    user = models.OneToOneField(User, unique=True, verbose_name=('user'))
    phone = models.CharField(max_length=20)


