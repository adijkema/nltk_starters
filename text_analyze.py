# ---- [ INFO ]--------------------------------------------------------------------------
# Show some use of the nltk library
# NLTK: Natural Language processing ToolKit; Bird, Steven, Edward Loper and Ewan Klein (2009), 
# Natural Language Processing with Python. O’Reilly Media Inc
# Site: https://www.nltk.org/
#
# pip install ntlk
# nltk needs to download wordbanks before using some functions. 
#   
# Goal: In several steps convert a text string into a grammatical tagged word list
# For use in python 2, add; 
#         from __future__ import print_function
# Python version: >= python 3 
# ----------------------------------------------------------------------------------------

# ---- [ INITIALIZE without all the IMPORTS ] --------------------------------------------
from os import system
_ = system('clear')
__pyversion__ = '3.0.0'
from time import time, localtime, perf_counter, asctime     # import only what is needed
from sys import version_info

import nltk
text = "Most users should install NLTK from a distribution. Please see the installation instructions. However, if you need an up-to-the-minute version, then you will have to install NLTK from the source repository. Once you've downloaded this, you'll need to run the top level setup.py program to install this version of NLTK on your machine."

def decorator_main(func):
    def wrap():
        if ".".join(map(str, version_info[:3])) < __pyversion__:
            print(f'Use Python vs {__pyversion__} for best result')
        starttime = perf_counter()
        print(f'[ Some use of the nltk library                       {asctime(localtime(time()))} ]')
        print(f'{"":-<81}\n\n')

        func()

        print(f'\n\n{"":-<81}')
        print(f'[ END                              code total time costs: {perf_counter() - starttime} ] \n\n')
    return wrap


# ---- [ CODE BLOCK ] --------------------------------------------------------------------------

@decorator_main
def main():

    print(f'String to be worked on:\n\'{text}\'')

    # break into sentences
    print(f'\nPrint string as broken down into sentences:')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    for e in sent_tokenize(text):
        print(f'- {e}')

    # tokenize to words
    print(f'\nPrint string broken down into words, all sentences, alphabetic order:')
    from nltk.tokenize import word_tokenize
    tokenized = word_tokenize(text)
    tokenized.sort()
    print(tokenized)

    # frequency distribution
    print(f'\nPrint for all the words its frequency distribution:')
    from nltk.probability import FreqDist
    fdist = FreqDist(tokenized)
    print(fdist)
    print(fdist.most_common(3))

    # import matplotlib.pyplot as plt 
    # fdist.plot(30,cumulative=False)
    # plt.show()

    # removing non-essential words via list of stopwords
    print(f'\nMake list of English stopwords, count and display them:')
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words("english"))  
    print(len(stopwords), '-', list(stopwords))

    print(f'\nClean list using stopwords, count cleaned words:')
    cleanedlist = []
    countclean = 0
    for e in tokenized:
        if e in stopwords:
            countclean += 1
        else:
            cleanedlist.append(e.lower())       
    print(countclean, '-', cleanedlist)

    # word normalisation, such as stemming. Not linguistical correct
    print(f'\nNormalize the cleaned list of words:')
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    stemmed_words = []
    for e in cleanedlist:
        stemmed_words.append(ps.stem(e))
    print(stemmed_words)

    # nltk.download('wordnet')
    print(f'\nLemmatize, make unique and sort the list of words:')
    from nltk.stem.wordnet import WordNetLemmatizer
    lem = WordNetLemmatizer()
    # from nltk.stem.porter import PorterStemmer
    # stem = PorterStemmer()
    wordnetted = []
    for e in cleanedlist:
        wordnetted.append(lem.lemmatize(e,"v"))
    words = list(set(wordnetted))
    words.sort()
    print(words)

    # POS tagging (Part-of-Speech) to identify grammatical groups of a given word, based on the context
    # nltk.download('averaged_perceptron_tagger')
    print(f'\nPOS tagging the list:')
    tokens = list(set(cleanedlist))
    tokens.sort()
    for e in nltk.pos_tag(tokens):
        print(e)

    print(f'\nIdentify named entities in all sentences:')
    for e in sent_tokenize(text):
        tokens = nltk.word_tokenize(e)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged)
        print(entities)
        

if __name__ == '__main__': main()

# ---- [ END CODE BLOCK ] --------------------------------------------------------------------------



