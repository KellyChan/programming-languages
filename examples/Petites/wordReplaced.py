def censor(text, word):
    return text.replace(word, "*"*len(word))
