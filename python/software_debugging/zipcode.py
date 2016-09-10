#
# Implement checkRep on this ZIPCode class
#
# Valid zip codes are 5 digits long, where each digit
# is between 0-9


class ZIPCode:
    # US Only
    
    def __init__(self, zip):
        self._zip = zip
        self.checkRep()

    def zip(self):
        return self._zip
    
    def checkRep(self):
        assert len(self.zip()) == 5
        # add in your asserts to ensure this is a valid ZIP code
        for i in range(0, 5):
            assert '0' <= self.zip()[i] <= '9'
