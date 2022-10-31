def char2int(char):
    if char == " ":
        return 27
    return ord(char) - ord('a') + 1


def int2char(num):
    if num == 27:
        return " "
    return chr(ord("a")+num-1)


def word_to_vec(word):
    vec = []
    for c in list(word):
        vec.append(char2int(c))
    return vec


def vec_to_word(vector):
    word = ""
    for v in vector:
        word = word + int2char(v)
    return word
