#!/usr/bin/python


def parseOutSubject(f):
    from nltk.stem.snowball import SnowballStemmer
    import string
  
    

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("Subject:",1)[1].split('\n',1)[0]
    words = ''
    if len(content) != 0:
        ### remove punctuation
        text_string = content.translate(string.maketrans("", ""), string.punctuation).split()

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        stemmer = SnowballStemmer('english')
        
        for word in text_string:
            word = word.strip()
            word = stemmer.stem(word)            
            words = words + ' ' + word
    else:
        pass


    return words

def parseOutBody(f):
    from nltk.stem.snowball import SnowballStemmer
    import string
  
    

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation).split()

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        stemmer = SnowballStemmer('english')
        
        for word in text_string:
            word = word.strip()
            word = stemmer.stem(word)            
            words = words + ' ' + word
    else:
        pass


    return words
