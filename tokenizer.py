import re

def tokenize(text: str) -> list:
    tokens = []
    #use regular expressions to slip line by any non English alphanumeric char (special, whitespace, etc)
    words = re.split(r'[^a-zA-Z0-9]+', text.lower().strip(), flags = re.ASCII)
    #this if else statement is instead of checking each word
    #gets rid of empty line token
    if len(words) == 1 and words[0] == '':
        pass
    #fixes if a line starts and ends with a special char
    elif words[0] == '' and words[len(words) - 1] == '':
        tokens += words[1:len(words) - 1]
    #fixes if a line starts with a special char
    elif words[0] == '':
        tokens += words[1:]
    #fixes if a line ends with a special char
    elif words[len(words) - 1] == '':
        tokens += words[:len(words) - 1]
    #normal add tokens on that line to total token list
    else:
        tokens += words
    return tokens
