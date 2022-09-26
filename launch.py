import json
import cProfile, pstats
from flask import Flask, render_template, request

import indexer
import search


def main():
    #uncomment line below to reform index
    #indexer.build_index()

    #for timing purposes, toggle timing var True of False to time query response or not
    timing = True

    #load the docid index and index of the index into memory (not the actual index)
    try:
        with open('docid_index.txt', 'r') as file:
            docid_index = json.load(file, object_hook=json_int)
        with open('index_index.txt', 'r') as file:
            index_index = json.load(file)
        with open('doc_lengths.txt', 'r') as file:
            doc_lengths = json.load(file, object_hook=json_int)
    except:
        print('Exception')


    #local GUI using flask
    app = Flask(__name__)

    #main home page GUI
    @app.route("/")
    def query():
        return render_template('query.html')

    #results page GUI
    @app.route('/results/', methods = ['POST', 'GET'])
    def results():
        if request.method == 'GET':
            return f"Error: /results is accessed directly, instead enter a query from the home page."
        if request.method == 'POST':
            q_form = request.form
            q = q_form['Query']

            if timing:
                profiler = cProfile.Profile()
                profiler.enable()

            #process the query
            url_list = search.process_query(q, docid_index, index_index, doc_lengths)

            urlstr_list = []
            n = 1
            for url in url_list:
                #k = 10 urls to show to the user
                if n > 10:
                    break
                urlstr_list.append(f'{n}) {url}')
                n += 1

            if timing:
                profiler.disable()
                stats = pstats.Stats(profiler).sort_stats('time')
                stats.print_stats()

            return render_template('results.html', urls=urlstr_list, initial_query=q)

    app.run(host='localhost', port=5000)


#turn str into int when json loading
def json_int(d: dict) -> dict:
    return {int(k):v for k,v in d.items()}


if __name__ == '__main__':
    main()
