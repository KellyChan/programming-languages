from nltk import wordpunct_tokenize


def train(language_stopwords, text):
    """
    Train the languages_ratios by checking how many stopwords in the text
    are in the language_stopwords.

        languages_ratios:
   
            {
                'english': set(['a', 'an']),
                'french': set(['un', 'une'])
            }

    @param language_stopwords:
    @type language_stopwords: dict

    @param text: Text that should be train
    @type text: str

    @return: a dict of languages_ratios against tex
    """

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    words_set = set(words)

    languages_ratios = {}
    for language in language_stopwords.keys():
        # get the stopwords set for this language
        stopwords_set = set(language_stopwords[language])
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements)

    return languages_ratios
