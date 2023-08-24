import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import ast


ELASTICSEARCH_URL = 'http://localhost:9200/'
FILE_PATH = 'sample-1M.jsonl'
NUM_DOCS_TO_INDEX = 100
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Aufgabe 1: Indexieren
# create index
def create_index(index_name, mappings):
    response = requests.request(method='PUT',
                                url=ELASTICSEARCH_URL + index_name,
                                json=mappings)
    print(response.json())


mappings = {
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "content": {"type": "text"},
      "content_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": True,
        "similarity": "l2_norm"
      }
    }
  }
}


# create vectors
def create_vectors_from_text_list(text_list):
    vector_list = []
    for text in text_list:
        vector_list.append((embedder.encode(text.lower(), convert_to_tensor=True)).tolist())
    return vector_list


# index documents
def index_docs(document_dicts):
    for idx, doc in enumerate(document_dicts):
        response = requests.request(method='POST',
                                    url=ELASTICSEARCH_URL + 'articles_vector_data/_create/_' + str(idx),
                                    json=doc)
        print(response.json())


# Aufgabe 2
def search_vector(input_query):
    input_embedding = (embedder.encode(input_query.lower(), convert_to_tensor=True)).tolist()

    query = {
          "knn": {
                "field": "content_vector",
                "query_vector": input_embedding,
                "k": 10,
                "num_candidates": 100
          },
          "fields": ["content"]
    }
    reponse = requests.request('GET', url=ELASTICSEARCH_URL + 'articles_vector_data/_search', json=query)
    print(reponse.json())


if __name__ == '__main__':

    # Aufgabe 1
    # Aufgabe 1a
    create_index(index_name='articles_vector_data', mappings=mappings)

    # Aufgabe 1b
    with open(FILE_PATH, 'r') as json_file:
        file_data = json_file.readlines()

    file_data = [ast.literal_eval(line) for line in file_data]
    text_list = [data['content'] for data in file_data]
    vecs = create_vectors_from_text_list(text_list[:NUM_DOCS_TO_INDEX])

    docs_to_index = []
    for idx, doc in enumerate(vecs):
        f_data = file_data[idx]
        f_data['content_vector'] = vecs[idx]
        docs_to_index.append(f_data)

    index_docs(docs_to_index)

    search_vector("Which artists are the headliners at Apple Music Festival?")



