import time


class TwoSum(object):

    def __init__(self, nums, target):
        self.nums = nums
        self.target = target


    def brute_force(self):
        """
        - time complexity: O(n^2)
        - space complexity: O(1)
        """

        time_start = time.time()

        for i in range(len(self.nums)):
            for j in range(i+1, len(self.nums), 1):
                if self.nums[i] + self.nums[j] == self.target:
                    time_end = time.time()
                    print "(Brute Force) Time Cost: %f" % (time_end - time_start)
                    return [i, j]

        time_end = time.time()
        print "(Brute Force) Time Cost: %f" % (time_end - time_start)
        return "No two sum solution."


    def two_pass_hash(self):
        """
        - time complexity: O(n)
        - space complexity: O(n)
        """

        time_start = time.time()
        # convert the num list as num dictionary
        num_dict = {}
        for (key, value) in enumerate(self.nums):
            num_dict[key] = value

        # look up (key, value) from the num dictionary
        for i in range(len(self.nums)):
            this_num = self.target - self.nums[i]
            keys = [key for key, value in num_dict.iteritems() if value == this_num]
            if keys and keys[0] != i:
                time_end = time.time()
                print "(Two Pass Hash) Time Cost: %f" % (time_end - time_start)
                return [i, keys[0]]

        time_end = time.time()
        print "(Two Pass Hash) Time Cost: %f" % (time_end - time_start)
        return "No two sum solution."


    def one_pass_hash(self):
        """
        - time complexity: O(n)
        - space complexity: O(n)
        """

        time_start = time.time()

        num_dict = {}
        for i in range(len(self.nums)):
            this_num = self.target - self.nums[i]
            if this_num in num_dict.values():
                keys = [key for key, value in num_dict.iteritems() if value == this_num]
                time_end = time.time()
                print "(One Pass Hash) Time Cost: %f" % (time_end - time_start)
                return [keys[0], i]
            num_dict[i] = self.nums[i]

        time_end = time.time()
        print "(One Pass Hash) Time Cost: %f" % (time_end - time_start)
        return "No two sum solution."          
