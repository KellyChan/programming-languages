import settings.settings as settings
from clientbase import FaceClientBase

class Detection(FaceClientBase):

    def __init__(self):
        super(Detection, self).__init__()

    def detect(self):
        faces = {name: self.api.detection.detect(url=url) for name, url in settings.PERSONS}
        return faces

    def landmark(self):
        pass
