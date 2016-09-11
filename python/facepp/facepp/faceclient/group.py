from clientbase import FaceClientBase

class Group(FaceClientBase):

    def __init__(self):
        super(Group, self).__init__()


    def create(self, group_name):
        return self.api.group.create(group_name=group_name)

    def delete(self, group_name):
        return self.api.group.delete(group_name=group_name)

    def add_person(self, group_name, person_name):
        """
        Returns:

            {'added': 3, 'success': True}
        """
        return self.api.group.add_person(group_name=group_name, person_name=person_name)

    def remove_person(self):
        pass

    def get_info(self):
        pass

    def set_info(self):
        pass

    def grouping(self):
        pass
