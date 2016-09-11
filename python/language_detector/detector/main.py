import corpus.stopwords
import train.stopwords


def detect_language(text, languages):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    @param text: Text whose language want to be detected
    @type text: str

    @param languages: the languages should be detected
    @type languages: list

    @return: Most scored language guessed
    @rtype: str
    """

    language_stopwords = corpus.stopwords.generate(languages)
    language_ratios = train.stopwords.train(language_stopwords, text)
    most_rated_language = max(language_ratios, key=language_ratios.get)

    return most_rated_language


if __name__ == '__main__':

    text = """Je suis Deni."""
    languages = ['english', 'french', 'german', 'spanish']

    language = detect_language(text, languages)
    print language
