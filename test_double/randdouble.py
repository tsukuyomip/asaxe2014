class RandDouble(object):
    """ Test Double
    rand(): 1.0 -> 0.0 -> 1.0 -> ..."""

    def __init__(self):
        self.nextval = 1.0
    def rand(self):
        retval = self.nextval
        self.nextval = abs(retval - 1.0)
        return retval
