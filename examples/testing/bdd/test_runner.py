if __name__ == "__main__":

    import nose
    from plugins.bddprinter import *

    nose.run(argv=["", "test_shoppingcart", "--with-bdd"], plugins=[BddPrinter()])

