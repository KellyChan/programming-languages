def anti_vowel(text):
    l = ""
    for i in range(0,len(text)):
        if text[i].lower() in "aeiou":
            pass
        else:
            l += text[i]
    return l
