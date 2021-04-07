import collections
from nltk.tokenize import RegexpTokenizer
from scipy import spatial

text = """Für diese Pflanze muss der Topf rund sein . Vor dem
        Einpflanzen der Setzlinge sollte die Erde flach getreten
        werden . Bei richtiger Pflege werden Sie schon innerhalb
        von wenigen Tagen beobachten können , wie sich die Blüte
        rund formt . Richtige Pflege bedeutet , mehrmals am Tag
        zu giessen , wobei der Einschlagswinkel des Wassers auf
        die Erde idealerweise flach sein sollte .
        Böse Zungen behaupten , dass die Erde flach ist . Die
        Wissenschaft hat bis anhin jedoch mehr oder weniger
        überzeugende Beweise geliefert , welche aufzeigen , dass
        unsere Welt mit grosser Wahrscheinlichkeit rund ist .
        Beweise , die für eine Erde sprechen , die flach ist ,
        sind jedoch meist gänzlich widerlegt ."""

def count_tokens(text, top):
    tokenizer = RegexpTokenizer(r'\w+') #Tokenizes
    tokens = tokenizer.tokenize(text)
    counter = collections.Counter(tokens)
    most_common = counter.most_common(top)
    return most_common

def context_window(text, top, context):
    c = [0 for x in range(9)]
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    counter = collections.Counter(tokens)
    most_common = counter.most_common(top)
    most_list = []
    x, y = 0, 0
    for i in most_common:   #Unnests most_common
        most_list.append(most_common[x][y])
        x += 1

    w = {"Erde": dict(zip(most_list, c)) , "flach": dict(zip(most_list, c)), "rund": dict(zip(most_list, c))}

    try:    #IndexError if it's longer or shorter than the list
        for i, word in enumerate(tokens):
            if word in w.keys():
                for j in range(-context, context+1):    #Goes through 5 words before and after the vector words
                    if tokens[i+j] in most_list:    #Checks whether the top words are in the context window and adds +1 if the case
                        w[word][tokens[i+j]] += 1
    except IndexError:
        pass
    return list(w["Erde"].values()), list(w["flach"].values()), list(w["rund"].values())

def cosine_similarity(text, top, context):
    Erde = context_window(text, top, context)[0] #Erde
    flach = context_window(text, top, context)[1] #flach
    rund = context_window(text, top, context)[2] #rund
    return 1 - spatial.distance.cosine(Erde, flach), 1 - spatial.distance.cosine(flach, rund), 1 - spatial.distance.cosine(rund, Erde)

if __name__ == '__main__':
    pass

print(cosine_similarity(text=text, top=8, context=5))