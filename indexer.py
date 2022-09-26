import json
import glob
import re
import math

import nltk
from bs4 import BeautifulSoup

import posting
import tokenizer


def build_index() -> None:
    #use glob to get all documents in the folder
    documents = glob.glob("../developer/DEV/**/*.json", recursive=True)
 
    #create all the dictionaries and other set up
    index = dict()
    docid_index = dict()
    index_index = dict()
    n = 0
    length_index = dict()
    posting_count = dict()
    min_tf = dict()
    min_tf_docid = dict()

    total_docs = len(documents)
    batch = []
    partial_index_names = ['partial_index_1.txt', 'partial_index_2.txt', 'partial_index_3.txt']
    partial_index_index_names = ['partial_index_index_1.txt', 'partial_index_index_2.txt', 'partial_index_index_3.txt']
    b = 0
    last_batch = False

    simhash_index = set()

    while len(documents) > 0:
        #set for 3 batches
        if b == 2:
            last_batch = True
        #get some of the docs and put in a batch
        batch, documents = get_batch(documents, total_docs, last_batch)

        #go through each document in the batch
        for doc in batch:
            n += 1
            with open(doc, 'r') as file:
                data = json.load(file)
            url = data['url']
            html = data['content']

            #defragment the url
            url_defragment = url.split('#', 1)[0]
        
            #use beautiful soup to parse the HTML (works with broken HTML)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()

            #tokenize the text
            tokens = tokenizer.tokenize(text)

            #process the tokens
            token_stems, token_freq = process_tokens(tokens)


            #check for duplicate pages
            simhash = calc_simhash(token_freq)
            if detect_duplicate(simhash, simhash_index, 0.95) == True:
                n -= 1
                continue
            else:
                simhash_index.add(simhash)

            #update tf score based on key text (bold, title, etc.)
            title_list = []
            for title in soup.find_all('title'):
                title_list.append(title.get_text())
                #5 extra weight for terms in the title
                important_text(title_list, 5, token_stems)

            bold_list = []
            for bold in soup.find_all('b'):
                bold_list.append(bold.get_text())
                #2 extra weight for terms in bold
                important_text(bold_list, 2, token_stems)

            strong_list = []
            for strong in soup.find_all('strong'):
                strong_list.append(strong.get_text())
                #2 extra weight for terms in strong
                important_text(strong_list, 2, token_stems)

            h1_list = []
            for h1 in soup.find_all('h1'):
                h1_list.append(h1.get_text())
                #4 extra weight for terms in h1
                important_text(h1_list, 4, token_stems)
            
            h2_list = []
            for h2 in soup.find_all('h2'):
                h2_list.append(h2.get_text())
                #3 extra weight for terms in h2
                important_text(h2_list, 3, token_stems)

            h3_list = []
            for h3 in soup.find_all('h3'):
                h3_list.append(h3.get_text())
                #3 extra weight for terms in h3
                important_text(h3_list, 3, token_stems)

            h4_list = []
            for h4 in soup.find_all('h4'):
                h4_list.append(h4.get_text())
                #3 extra weight for terms in h4
                important_text(h4_list, 3, token_stems)

            h5_list = []
            for h5 in soup.find_all('h5'):
                h5_list.append(h5.get_text())
                #3 extra weight for terms in h5
                important_text(h5_list, 3, token_stems)

            h6_list = []
            for h6 in soup.find_all('h6'):
                h6_list.append(h6.get_text())
                #3 extra weight for terms in h6
                important_text(h6_list, 3, token_stems)

            length_index[n] = 0
            #map/update the token to list of postings
            for k,tf in token_stems.items():    
                if k not in index:
                    index[k] = list()
                    index[k].append(posting.Posting(n, tf))
                    posting_count[k] = 1
                    min_tf[k] = tf
                    min_tf_docid[k] = n
                elif posting_count[k] < 100:
                    index[k].append(posting.Posting(n, tf))
                    posting_count[k] += 1
                    if tf < min_tf[k]:
                        min_tf[k] = tf
                        min_tf_docid[k] = n
                else:
                    #if more than 100 postings (in a batch) for the term, trim off the lowest frequency posting
                    if tf > min_tf[k]:
                        index[k].remove(posting.Posting(min_tf_docid[k], min_tf[k]))
                        
                        new_min_tf_docid, new_min_tf = calc_trim_posting(n, tf, index[k])
                        min_tf[k] = new_min_tf
                        min_tf_docid[k] = new_min_tf_docid

                        index[k].append(posting.Posting(n, tf))

                #calculate lengths of documents here to make it faster to computer cosine simularity during search
                tf_log = 1 + math.log10(tf)
                length_index[n] += tf_log**2

            #update the docID index
            docid_index[n] = url_defragment

        #write the indexes to files
        write_to_file(index, index_index, 0, partial_index_names[b], partial_index_index_names[b])

        b += 1

        #reset these indexes after each batch of docs
        index = dict()
        index_index = dict()
        posting_count = dict()
        min_tf = dict()
        min_tf_docid = dict()

    #merge the partial index files into one index file
    merge_partial_indexes(partial_index_names[0], partial_index_index_names[0], partial_index_names[1], partial_index_index_names[1], 'partial_index_1and2.txt', 'partial_index_index_1and2.txt')
    merge_partial_indexes(partial_index_names[2], partial_index_index_names[2], 'partial_index_1and2.txt', 'partial_index_index_1and2.txt', 'index.txt', 'index_index.txt')

    #dump docids to a file
    with open('docid_index.txt', 'w') as file:
        json.dump(docid_index, file)

    #dump doc lengths to a file
    for k,v in length_index.items():
        length_index[k] = math.sqrt(v)
    with open('doc_lengths.txt', 'w') as file:
        json.dump(length_index, file)
    
    #keep track of extra stats
    with open('stats.txt', 'w') as file:
        file.write(f'Number of docs: {n}')



def process_tokens(tokens: str) -> dict:
    #use nltk to stem the tokens
    stemmer = nltk.stem.PorterStemmer()
    token_stems = dict()
    token_freq = dict()

    for token in tokens:
        t = stemmer.stem(token)
        #calculate frequency
        if t not in token_stems:
            token_stems[t] = 1
        else:
            token_stems[t] += 1
        if token not in token_freq:
            token_freq[token] = 1
        else:
            token_freq[token] += 1

    return token_stems, token_freq


def write_to_file(index: dict, index_index: dict, counter: int, index_name: str, index_index_name: str) -> None:
    #write each term to file line by line and keep track of an index of the index
    with open(index_name, 'w') as file:
        for k,v in sorted(index.items(), key=lambda x: x[0]):
            #create a line from the term and posting
            line = ''
            line += k
            for p in v:
                line += p.write()
            line += '\n'    
            file.write(line)
            #update the index of the index with the correct seek number
            index_index[k] = counter + len(k)
            counter += len(line) + 1

    #dump the index of the index to a file
    with open(index_index_name, 'w') as file:
        json.dump(index_index, file)


def get_batch(documents: list, total_docs: int, last_batch: bool) -> list:
    #split the documents into 3 batches
    batch_docs = int(total_docs/3)
    if last_batch:
        return documents, []
    return documents[:batch_docs], documents[batch_docs:]


def merge_partial_indexes(i1_name: str, ii1_name: str, i2_name: str, ii2_name: str, destination_index_name: str, destination_index_index_name: str) -> None:
    index_index = dict()

    #open both files
    with open(ii1_name, 'r') as file:
            ii1 = json.load(file)
    with open(ii2_name, 'r') as file:
            ii2 = json.load(file)

    i1_terms = sorted(ii1.keys())
    i2_terms = sorted(ii2.keys())

    i = 0
    j = 0
    counter = 0

    with open(i1_name, 'r') as file1, open(i2_name, 'r') as file2, open(destination_index_name, 'w') as file3:
        while i < len(i1_terms) and j < len(i2_terms):
            if i1_terms[i] == i2_terms[j]:
                #merge and add postings from both
                seek_pos1 = ii1[i1_terms[i]]
                file1.seek(seek_pos1)
                posting_line1 = file1.readline()
                posting_list1 = get_posting_list(posting_line1)
                seek_pos2 = ii2[i2_terms[j]]
                file2.seek(seek_pos2)
                posting_line2 = file2.readline()
                posting_list2 = get_posting_list(posting_line2)

                m = 0
                n = 0
                posting_list = []
                #walk through both lists
                while m < len(posting_list1) and n < len(posting_list2):
                    if posting_list1[m].get_docid() < posting_list2[n].get_docid():
                        posting_list.append(posting_list1[m])
                        m += 1
                    else:
                        posting_list.append(posting_list2[n])
                        n += 1
                while m < len(posting_list1):
                    posting_list.append(posting_list1[m])
                    m += 1
                while n < len(posting_list2):
                    posting_list.append(posting_list2[n])
                    n += 1

                line = ''
                line += i1_terms[i]
                for p in posting_list:
                    line += p.write()
                line += '\n'    
                file3.write(line)
                index_index[i1_terms[i]] = counter + len(i1_terms[i])
                counter += len(line) + 1

                i += 1
                j += 1
            else:
                if i1_terms[i] < i2_terms[j]:
                    #add only term and postings from i1
                    seek_pos = ii1[i1_terms[i]]
                    file1.seek(seek_pos)
                    posting_line = file1.readline()
                    posting_list = get_posting_list(posting_line)

                    line = ''
                    line += i1_terms[i]
                    for p in posting_list:
                        line += p.write()
                    line += '\n'    
                    file3.write(line)
                    index_index[i1_terms[i]] = counter + len(i1_terms[i])
                    counter += len(line) + 1

                    i += 1
                else:
                    #add only term and postings from i2
                    seek_pos = ii2[i2_terms[j]]
                    file2.seek(seek_pos)
                    posting_line = file2.readline()
                    posting_list = get_posting_list(posting_line)

                    line = ''
                    line += i2_terms[j]
                    for p in posting_list:
                        line += p.write()
                    line += '\n'    
                    file3.write(line)
                    index_index[i2_terms[j]] = counter + len(i2_terms[j])
                    counter += len(line) + 1

                    j += 1

    #dump the updated index of the index to a file
    with open(destination_index_index_name, 'w') as file:
        json.dump(index_index, file)


def get_posting_list(posting_line: str) -> list:
    #parsing the index text format of postings list
    postings_str = re.split('[\(\)]+', posting_line)
    p_list = []
    for p_str in postings_str:
        if p_str != '' and p_str != '\n':
            post = p_str.split(',')
            p = posting.Posting(int(post[0]), int(post[1]))
            p_list.append(p)
    return p_list


def calc_simhash(token_weights: list) -> str:
    #defining weights as the frequency of the tokens
    binary = dict()
    #get the 32 bit binary hash for each token
    for k in token_weights.keys():
        binary[k] = hash32b(k)
    bit = 31
    fingerprint = ''
    #go through each bit of each hashed token and calculate the fingerprint based on weights
    while bit >= 0:
        vec = 0
        for k,v in binary.items():
            if v[bit] == '0':
                vec -= token_weights[k]
            else:
                vec += token_weights[k]
        if vec > 0:
            fingerprint = '1' + fingerprint
        else:
            fingerprint = '0' + fingerprint
        bit -= 1
    return fingerprint

def hash32b(token: str) -> str:
    #use the default python hash function
    hashint = hash(token)
    b = 2**32
    #mod it so can be represented by 32 bits
    hashbin = f'{hashint % b:b}'
    #add extra zeros to the front if neccessary
    while len(hashbin) < 32:
        hashbin = '0' + hashbin
    return hashbin


def detect_duplicate(simhash: str, simhash_index: set, threshold: float) -> bool:
    s1 = int(simhash, 2)
    #compare the given simhash to all unique simhashes found before
    for shash in simhash_index:
        #calculate similar bits in the fingerprints
        s2 = int(shash, 2)
        i = s1 ^ s2
        diff = 0.0
        while i:
            i = i & (i-1)
            diff += 1.0
        #using 32 bit fingerprints
        similar = 32.0 - diff
        #if the number of similar bits divided by total bits is greater than the given threshold, return true
        if similar/32.0 >= threshold:
            return True
    return False


def important_text(word_list: list, value: int, token_index: dict):
    for words in word_list:
        tokens = tokenizer.tokenize(words)
        for token in tokens:
            stemmer = nltk.stem.PorterStemmer()
            t = stemmer.stem(token)
            if  t not in token_index:
                token_index[t] = value
            else:
                token_index[t] += value

    
def calc_trim_posting(n: int, tf: int, posting_list: list) -> int:
    new_min_tf = tf
    new_min_tf_docid = n
    for p in posting_list:
        if p.get_frequency() < new_min_tf:
            new_min_tf = p.get_frequency()
            new_min_tf_docid = p.get_docid()
    return new_min_tf_docid, new_min_tf
