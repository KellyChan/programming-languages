import urllib

def get_param(param_value):
    """Return the nothing_value from the specific url
    """

    param = urllib.urlencode({'nothing': param_value})
    url = "http://www.pythonchallenge.com/pc/def/linkedlist.php?%s" % param
    content = urllib.urlopen(url).read()
    # split the content by " " and return the last word
    value = content.split(" ")[-1]

    return value


if __name__ == '__main__':

    init_value = "12345"
    param = get_param(init_value)
    while param > 0:
        print param
        param = get_param(param)
