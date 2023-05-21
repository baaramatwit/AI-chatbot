from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Want to split our sentence into array of words and tokens 
    a token can be a word or punctuation , or number 
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stem - find the root form of the word 
    examples: "programming," "programmer," "programs" reduced down 
    to the common word stem "program".
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Example what we need to do:

        sentence = ["hello" "how" "are" "you"] - tokenized sentence
    words = ["hi", "hello" "I" "you" "bye" "thank" "cool"] - look at each word in sentence, if avilable in words sentence we give 1 at precesion where located
    bag = [ 0 , 1 , 0, 1, ,0, 0, 0]
    """
    # use list comprehension to stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # inititalize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0  # if in sentence gets a 1

    return bag
