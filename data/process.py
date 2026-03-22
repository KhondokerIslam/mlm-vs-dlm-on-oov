import pandas as pd

from spellchecker import SpellChecker

spell = SpellChecker(distance=2)

def read_tsv( tsv_path ):

    data = pd.read_csv(tsv_path, sep = "\t", names = ['tweet', 'label'] )

    print( "Data Loaded!" )

    return data

def spell_check(data):

    def correct_sentence(s):
        tokens = []
        for word in s.split():

            if word.startswith("#") or word.startswith("@"):
                tokens.append(word)
                continue

            if word.lower() in spell:
                tokens.append(word)
            else:
                tokens.append(spell.correction(word) or word)

        sentence = " ".join(tokens)
        return sentence

    data["tweet"] = data["tweet"].apply(correct_sentence)

    ## Debuging
    # data.loc[0, "tweet"] = correct_sentence(data.loc[0, "tweet"])


    print( "Spelling Fixed!" )

    return data


def save_tsv( data, loc = "" ):

    data.to_csv(loc, sep="\t", index = False, header = None)

    print( "Data Saved!" )

    return None


if "__main__":

    data = read_tsv( "../dataset/test.tsv" )

    data = spell_check( data )

    save_tsv( data, loc = "../dataset/nltk.tsv")