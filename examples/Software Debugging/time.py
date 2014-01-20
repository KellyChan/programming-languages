#
# Change the Time class so that hours, minutes and seconds 
# are stored as ints
#

class Time:
    def __init__(self, h = 0, m = 0, s = 0):
        assert 0 <= int(h) <= 23
        assert 0 <= int(m) <= 59
        assert 0 <= int(s) <= 60
        
        self._hours   = int(h)
        self._minutes = int(m)
        self._seconds = int(s)
                
    def hours(self):
        return self._hours

    def minutes(self):
        return self._minutes

    def seconds(self):
        return self._seconds

    def __repr__(self):
        return "{0:02d}:{1:02d}:{2:02d}".format(
            self.hours(), self.minutes(), self.seconds())

def test():
    t = Time(3.14, 0, 0)
    assert repr(t) == "03:00:00"



