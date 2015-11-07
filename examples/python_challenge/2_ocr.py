# dictionary: http://www.math.sjsu.edu/~foster/dictionary.txt
# download the dictionary for mapping the words

def print_sorted_keys(dictionary):
    print sorted(dictionary.items(), key=lambda dictionary: dictionary[0])

def print_sorted_values(dictionary):
    print sorted(dictionary.items(), key=lambda dictionary: dictionary[1])

def generate_word_dict(file_path):

    word_counts = {}

    text = open(file_path, 'r')
    for line in text.readlines():
        for char in line:
            if char not in word_counts.keys():
                word_counts[char] = 1
            else:
                word_counts[char] += 1
    text.close()

    return word_counts

def mapping(letters, dictbook):

    words = []

    dictbook = open(dictbook, 'r')
    for line in dictbook.readlines():
        # if set(line.strip()) == set(letters):
        #     words.append(line.strip())
        words.append(line.strip()) if set(line.strip()) == set(letters) else words
    dictbook.close()

    return words

if __name__ == '__main__':

    file_path = './data/2_ocr.txt'
    dictbook = "./data/dictionary.txt" 

    # find the letters
    word_counts = generate_word_dict(file_path)
    print_sorted_keys(word_counts)
    print_sorted_values(word_counts)

    # look for a word from a dictionary
    letters = "aeilquty" 
    words = mapping(letters, dictbook) 
    print words
