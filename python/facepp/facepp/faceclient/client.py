"""
    FaceClient
       \---- Info
                \---- FaceClientBase
       \---- Detection
                \---- FaceClientBase
       \---- Faceset
                \---- FaceClientBase
       \---- Person
                \---- FaceClientBase
       \---- Group
                \---- FaceClientBase
       \---- Train
                \---- FaceClientBase
       \---- Recognition
                \---- FaceClientBase
"""

from info import Info

from detection import Detection
from faceset import Faceset
from person import Person
from group import Group

from train import Train
from recognition import Recognition

class FaceClient(object):

    def __init__(self):
        self.info = Info()

        self.detection = Detection()
        self.faceset = Faceset()
        self.person = Person()
        self.group = Group()

        self.train = Train()
        self.recognition = Recognition()  
