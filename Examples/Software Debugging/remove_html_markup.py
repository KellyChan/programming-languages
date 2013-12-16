# 
# Modify `remove_html_markup` so that it actually works!
#

def remove_html_markup(s):
    tag   = False
    quote = False
    out   = ""

    for c in s:
        # print c, tag, quote
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag and not quote:
            quote = not quote
        elif not tag:
            out = out + c

    return out


def test():
    assert remove_html_markup('<a href="don' + "'" + 't!">Link</a>') == "Link"
