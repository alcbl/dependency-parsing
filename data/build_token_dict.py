import spacy

nlp = spacy.load('fr')
STOP_WORDS = spacy.lang.fr.stop_words.STOP_WORDS


def get_token_from_text(text, is_remove_stop_words=True):
    doc = nlp(text)
    if is_remove_stop_words:
        return [token.lemma_.lower() for token in doc if not token.is_stop]
    else:
        return [token.lemma_.lower() for token in doc]

def extract_token_from_text_file(filename):
    token = []
    with open(filename, "r") as f:
        for line in f.readlines():
            elements = line.split("\t")
            if len(elements) == 3:
                token += get_token_from_text(elements[2].strip("\n"))
    return token

def get_more_frequent_token(token, num_token=100):
    frequency_dict = {}
    for tok in token:
        if tok in frequency_dict.keys():
            frequency_dict[tok] += 1
        else:
            frequency_dict[tok] = 1
    sorted_frequency_items = sorted(frequency_dict.items(),
                                    key=lambda item: item[1],
                                    reverse=True)
    return [item[0] for item in sorted_frequency_items[:num_token]]

token = extract_token_from_text_file("train.text")
token = get_more_frequent_token(token, 150)

with open("token.dict", "w") as f:
    for tok in token:
        f.write("{}\n".format(tok))
