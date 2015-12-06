import unittest

from mock import Mock
from copy import deepcopy

from shoppingcart import *

class CartThatWeWillSaveAndRestoreUsingVoidspaceMock(unittest.TestCase):


    def test_fill_up_a_cart_then_save_it_and_restore_it(self):

        # create an empty shopping cart
        cart = ShoppingCart(DataAccess())

        # add a couple of items
        cart.add("carton of milk", 2.50)
        cart.add("forzen pizza", 3.00)
        self.assertEquals(2, len(cart))

        # create a clone of the cart for mocking purposes
        original_cart = deepcopy(cart)

        # save the cart at this point in time into a database
        # using a mock
        cart.storer.store_cart = Mock()
        cart.storer.store_cart.return_value = 1
        cart.storer.retrieve_cart = Mock()
        cart.storer.retrieve_cart.return_value = original_cart
        id = cart.store()
        self.assertEquals(1, id)

        # add more items to cart
        cart.add("cookie dough", 1.75)
        cart.add("ginger ale", 3.25)
        self.assertEquals(4, len(cart))

        # restore the cart to the last point in time
        cart.restore(id)
        self.assertEquals(2, len(cart))
        cart.storer.store_cart.assert_called_with(cart)
        cart.storer.retrieve_cart.assert_called_with(1)

