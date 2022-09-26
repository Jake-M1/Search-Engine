import nltk
import math

import tokenizer
import posting
import indexer


def process_query(query: str, docid_index: dict, index_index: dict, doc_lengths: dict) -> str:
    #tokenize and stem the query the same as the docs
    tokens = tokenizer.tokenize(query)
    stemmer = nltk.stem.PorterStemmer()
    term_list = []
    for token in tokens:
        t = stemmer.stem(token)
        term_list.append(t)

    #merging duplicates and calculating query term frequency before going through the postings
    query_dict = dict()
    for token in tokens:
        t = stemmer.stem(token)
        if t in query_dict:
            query_dict[t] += 1
        else:
            query_dict[t] = 1
    query_tf_weights = dict()
    #log the query term frequncy
    for k,v in query_dict.items():
        query_tf_weights[k] = 1 + math.log10(v)


    #keep an accumulator of docids and scores
    accumulator = dict()
    n = len(docid_index)
    query_tf_idf = dict()

    #term at a time retreival
    with open('index.txt') as file:
        for term,q_tf in query_tf_weights.items():
            if term in index_index:
                #use undex of index to seek to the right position in the index file and get postings fast
                seek_pos = index_index[term]
                file.seek(seek_pos)
                posting_line = file.readline()
                posting_list = indexer.get_posting_list(posting_line)
                
                #calculate inverse document frequency for weighting
                df = len(posting_list)
                idf = math.log10(n/df)

                #query tf-idf score
                query_tf_idf[term] = q_tf * idf

                #update accumulator through multiple terms
                for p in posting_list:
                    #calculate document term frequency for weighting
                    d_tf = 1 + math.log10(p.get_frequency())

                    #normalize the document tf weights
                    if p.get_docid() not in accumulator:
                        accumulator[p.get_docid()] = dict()
                    accumulator[p.get_docid()][term] = d_tf/doc_lengths[p.get_docid()]

    #keep two different dictionaries for docs containing all the query terms and docs that do not
    scores = dict()
    scores_all_terms = dict()
    for d_id, d_norm in accumulator.items():
        score, all_terms = cosine_similarity(query_tf_idf, d_norm)
        if all_terms == True:
            scores_all_terms[d_id] = score
        else:
            scores[d_id] = score

    urls = []
    #return urls with decreasing cosine simularity score weights (so probably most relevant in the front)
    #first get the urls with all the query terms in them
    for docid,score in sorted(scores_all_terms.items(), key = lambda x: x[1], reverse = True):
        urls.append(docid_index[docid])

    num_urls = len(urls)
    #if not enough urls, pick from the highest scores urls that do not contain all query terms
    if num_urls < 10:
        for docid,score in sorted(scores.items(), key = lambda x: x[1], reverse = True):
            urls.append(docid_index[docid])
            num_urls += 1
            if num_urls >= 10:
                break

    return urls


def cosine_similarity(q: dict, d_norm: dict):
    #cosine simularity for lnc.ltc
    q_length = 0.0
    #calulate the lengthfor the query vector
    for tf_idf in q.values():
        q_length += tf_idf
    q_length = math.sqrt(q_length)

    #calculate query normalized weights
    q_norm = dict()
    for k,v in q.items():
        q_norm[k] = v/q_length

    #keep track if all the query terms are in the document
    all_terms = True

    score = 0
    #dot product each corresponding query normalized weight and document normalized weight for each term by multiplying and the adding them all up
    for term, q_tfidf_norm in q_norm.items():
        if term in d_norm:
            d_tf_norm = d_norm[term]
        else:
            d_tf_norm = 0
            all_terms = False
        score += q_tfidf_norm * d_tf_norm

    return score, all_terms

