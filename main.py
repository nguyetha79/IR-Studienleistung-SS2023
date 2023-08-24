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
            "k": 50,
            "num_candidates": 100
        },
        "fields": ["content"],
        "size": 50
    }
    reponse = requests.request('GET', url=ELASTICSEARCH_URL + 'articles_vector_data/_search', json=query)
    return reponse.json()


# Aufgabe 3

def evaluation(index_name, num_eval_docs, ground_truth_list, predictions_list):
    print("--------------------")
    for x in range(num_eval_docs):
        print(f'{index_name}{x}')
        print('Precision@5:', precision(ground_truth_list[x], predictions_list[x], k=5))
        print('Recall@5:', recall(ground_truth_list[x], predictions_list[x], k=5))
        print('Precision@10:', precision(ground_truth_list[x], predictions_list[x], k=10))
        print('Recall@10:', recall(ground_truth_list[x], predictions_list[x], k=10))


def precision(true, pred, k=5):
    return len(set(true).intersection(set(pred[:k]))) / len(set(pred[:k]))


def recall(true, pred, k=5):
    return len(set(true).intersection(set(pred[:k]))) / len(set(true))


def get_relevant_docs_ids_list(json_response):
    ids = []
    for row in json_response["hits"]["hits"]:
        ids.append(row["_source"]["id"])

    return ids


def search_index(input_query, index_name):
    query = {
        "size": 50,
        "from": 0,
        "query": {
            "match": {
                "title": input_query
            }
        }
    }

    reponse = requests.request('GET', url=ELASTICSEARCH_URL + index_name + '/_search', json=query)
    return reponse.json()


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

    # Aufgabe 2
    res = search_vector("When did Edward Snowden write his first Twitter post?")
    print(res)

    # Aufgabe 3
    # Aufgabe 3a

    predictions_standard_0 = get_relevant_docs_ids_list(search_index(input_query="Who bought National Geographic magazine?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_0 = []

    predictions_standard_1 = get_relevant_docs_ids_list(search_index(input_query="What are people fleeing from in Syria?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_1 = []

    predictions_standard_2 = get_relevant_docs_ids_list(search_index(input_query="How much does the iPad Pro cost?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_2 = []

    predictions_standard_3 = get_relevant_docs_ids_list(search_index(input_query="What was discovered on Mars?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_3 = []

    predictions_standard_4 = get_relevant_docs_ids_list(search_index(input_query="How many people flee to Germany?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_4 = []

    predictions_standard_5 = get_relevant_docs_ids_list(
        search_index(input_query="Which artists are the headliners at Apple Music Festival?",
                     index_name="articles_10000_data"))
    ground_truth_standard_5 = []

    predictions_standard_6 = get_relevant_docs_ids_list(
        search_index(input_query="When did Edward Snowden write his first Twitter post?",
                     index_name="articles_10000_data"))
    ground_truth_standard_6 = []

    predictions_standard_7 = get_relevant_docs_ids_list(
        search_index(input_query="Which country launched its first space observatory into space?",
                     index_name="articles_10000_data"))
    ground_truth_standard_7 = []

    predictions_standard_8 = get_relevant_docs_ids_list(search_index(input_query="What is Chancellor Merkl asking of Facebook?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_8 = []

    predictions_standard_9 = get_relevant_docs_ids_list(search_index(input_query="Why went Facebook down?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_9 = []

    # Aufgabe 3b

    predictions_process_0 = get_relevant_docs_ids_list(search_index(input_query="Who bought National Geographic magazine?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_0 = []

    predictions_process_1 = get_relevant_docs_ids_list(search_index(input_query="What are people fleeing from in Syria?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_1 = []

    predictions_process_2 = get_relevant_docs_ids_list(search_index(input_query="How much does the iPad Pro cost?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_2 = []

    predictions_process_3 = get_relevant_docs_ids_list(search_index(input_query="What was discovered on Mars?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_3 = []

    predictions_process_4 = get_relevant_docs_ids_list(search_index(input_query="How many people flee to Germany?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_4 = []

    # Evaluation

    predictions_standard = [predictions_standard_0, predictions_standard_1, predictions_standard_2,
                            predictions_standard_3, predictions_standard_4]

    ground_truth_standard = [ground_truth_standard_0, ground_truth_standard_1, ground_truth_standard_2,
                             ground_truth_standard_3, ground_truth_standard_4]

    evaluation(index_name="\nArticles 10000 data standard ",
               num_eval_docs=5,
               ground_truth_list=ground_truth_standard,
               predictions_list=predictions_standard)

    predictions_process = [predictions_process_0, predictions_process_1, predictions_process_2,
                           predictions_process_3, predictions_process_4]

    ground_truth_process = [ground_truth_process_0, ground_truth_process_1, ground_truth_process_2,
                            ground_truth_process_3, ground_truth_process_4]

    evaluation(index_name="\nArticles 10000 data process ",
               num_eval_docs=5,
               ground_truth_list=ground_truth_process,
               predictions_list=predictions_process)

    # Aufgabe 3c
    predictions_100_standard_0 = get_relevant_docs_ids_list(
        search_index(input_query="Which artists are the headliners at Apple Music Festival?",
                     index_name="articles_100_data"))
    ground_truth_100_standard_0 = []

    predictions_100_standard_1 = get_relevant_docs_ids_list(
        search_index(input_query="When did Edward Snowden write his first Twitter post?",
                     index_name="articles_100_data"))
    ground_truth_100_standard_1 = []

    predictions_100_standard = [predictions_100_standard_0, predictions_100_standard_1]
    ground_truth_100_standard = [ground_truth_100_standard_0, ground_truth_100_standard_1]

    evaluation(index_name="\nArticles 100 data standard ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_standard,
               predictions_list=predictions_100_standard)

    predictions_100_process_0 = get_relevant_docs_ids_list(
        search_index(input_query="Which artists are the headliners at Apple Music Festival?",
                     index_name="processed_100_data"))
    ground_truth_100_process_0 = []

    predictions_100_process_1 = get_relevant_docs_ids_list(
        search_index(input_query="When did Edward Snowden write his first Twitter post?",
                     index_name="processed_100_data"))
    ground_truth_100_process_1 = []

    predictions_100_process = [predictions_100_process_0, predictions_100_process_1]
    ground_truth_100_process = [ground_truth_100_process_0, ground_truth_100_process_1]

    evaluation(index_name="\nArticles 100 data process ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_process,
               predictions_list=predictions_100_process)

    predictions_100_vector_0 = get_relevant_docs_ids_list(
        search_vector(input_query="Which artists are the headliners at Apple Music Festival?"))
    ground_truth_100_vector_0 = ['06703828-aec2-4e5a-bb6f-8f7aa4fb9357']

    predictions_100_vector_1 = get_relevant_docs_ids_list(
        search_vector(input_query="When did Edward Snowden write his first Twitter post?"))
    ground_truth_100_vector_1 = []

    predictions_100_vector = [predictions_100_vector_0, predictions_100_vector_1]
    ground_truth_100_vector = [ground_truth_100_vector_0, ground_truth_100_vector_1]

    evaluation(index_name="\nArticles 100 data vector ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_vector,
               predictions_list=predictions_100_vector)


