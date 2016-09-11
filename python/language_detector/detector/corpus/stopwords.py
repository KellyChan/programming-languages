from nltk.corpus import stopwords


def generate(languages):
    """
    Generate a dict of language stopwords from nltk.corpus by the specific
    lanauges.

        language_stopwords:
             {
                {'english': {'a', 'an'},
                {'french': {'un', 'une'}
             }

    @param languages: languages specified, e.g. ['english', 'french']
    @type languages: list

    @return: a dict of stopwords corresponding to the specific languages
    @rtype: dict
    """

    language_stopwords = {}

    for language in stopwords.fileids():
        if language in languages:
            stopwords_set = set(stopwords.words(language))
            language_stopwords[language] = stopwords_set

    return language_stopwords
