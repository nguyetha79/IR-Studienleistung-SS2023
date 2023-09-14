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
    print("\n--------------------")
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

    predictions_standard_0 = get_relevant_docs_ids_list(
        search_index(input_query="What are the problems caused by domestic violence?",
                     index_name="articles_10000_data"))
    ground_truth_standard_0 = ["4797df42-b7bd-46ef-bde5-b7a6b77cbb0e", "c5e25672-812b-43a0-b558-9cf89f4f1c3d",
                               "0d25ebac-299e-47d2-a76b-03e3cb069222", "3d7ca512-e7c7-4ea2-97ea-3d4885f7583f",
                               "49a6740c-8143-41df-b90f-63d6ddcc4804", "a92c78dd-48fe-4752-9a70-dba27e606288"]

    predictions_standard_1 = get_relevant_docs_ids_list(
        search_index(input_query="What are people fleeing from in Syria?",
                     index_name="articles_10000_data"))
    ground_truth_standard_1 = ["8402fd9a-10a9-406b-8a6e-57d295388522", "4caf2601-4da7-4b13-8b79-ea57a6ef7e9b",
                               "7d5db63e-efe3-4fd3-917c-7f9682b861c9", "cfccd4c7-0898-41cb-9f84-4d11267968fa",
                               "b76fbe7d-c648-49ef-9fd4-6d7759194e11", "30d58186-6f13-423d-9616-de3c6be5e8b3",
                               "eb7b1db2-53c6-4275-941a-35814e859c57", "780ece8a-7dde-44b6-b2bf-801c7d85b6da",
                               "a1ce1914-3129-4e10-b466-405639d27dee", "c6083819-8fd7-4359-b355-2fc7d3204155",
                               "402f6983-3300-4546-82c5-6f4eb08d0ba7"]

    predictions_standard_2 = get_relevant_docs_ids_list(search_index(input_query="What was discovered on Mars?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_2 = ["1eba3050-1cd2-4184-be76-bbe6f0b8f024", "5a07213b-7e81-4e0c-9dfa-758ae4c1ae3c",
                               "13ebe973-e727-47d5-8fe7-a27e05c2d578", "33ede757-4c7e-44ad-86be-c7401061af32",
                               "26a92e12-f6da-45be-afc9-4f13f699196c", "f0e7b753-2351-497a-a861-75ca339432c8"]

    predictions_standard_3 = get_relevant_docs_ids_list(
        search_index(input_query="Which artists are the headliners at Apple Music Festival?",
                     index_name="articles_10000_data"))
    ground_truth_standard_3 = ["0a452c76-196d-45d9-b104-09ba91570f1c", "7665c596-5e5f-4fdb-8ac8-eaa0bec7f4e9",
                               "781ec5d1-c8d4-46a5-ac94-5786c65c48fb"]

    predictions_standard_4 = get_relevant_docs_ids_list(
        search_index(input_query="Which country launched its first space observatory into space?",
                     index_name="articles_10000_data"))
    ground_truth_standard_4 = ["015eb5ca-1db1-4f09-adfc-58ad0ac3e456", "44611997-6d4b-4070-ba84-2cd46184b5b6"]

    predictions_standard_5 = get_relevant_docs_ids_list(search_index(input_query="How much does the iPad Pro cost?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_5 = ["f60ae2c2-9e98-404e-96ba-446bab8f16f5", "f9c85862-c318-49f8-8a83-c4cc8ca26c77",
                               "350b2f99-ce3a-40f9-8592-31937f31636b", "c3492f85-dfe1-4eda-864a-f8bab3b4e7ff",
                               "402f6983-3300-4546-82c5-6f4eb08d0ba7"]

    predictions_standard_6 = get_relevant_docs_ids_list(search_index(input_query="How many people flee to Germany?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_6 = ["848dcf97-c5e2-4df5-835d-5dfe56518b12", "1c85ed48-6172-488a-9d72-5e886d0b0e3e",
                               "f68e5f7b-5efe-4456-99ff-918357866e94", "4caf2601-4da7-4b13-8b79-ea57a6ef7e9b",
                               "8c1392a2-c8c3-4337-8bf5-2b9bcc4680c0", "7d13230a-0fb6-4d09-9a27-1b7df301c7a3",
                               "1f5a8552-460e-46e9-9065-796c3daf9df1"]

    predictions_standard_7 = get_relevant_docs_ids_list(
        search_index(input_query="When did Edward Snowden write his first Twitter post?",
                     index_name="articles_10000_data"))
    ground_truth_standard_7 = []

    predictions_standard_8 = get_relevant_docs_ids_list(
        search_index(input_query="Who bought National Geographic magazine?",
                     index_name="articles_10000_data"))
    ground_truth_standard_8 = ["e1aebec4-52d5-4ce9-9783-24751c27706b"]

    predictions_standard_9 = get_relevant_docs_ids_list(search_index(input_query="Why went Facebook down?",
                                                                     index_name="articles_10000_data"))
    ground_truth_standard_9 = []

    # Aufgabe 3b

    predictions_process_0 = get_relevant_docs_ids_list(
        search_index(input_query="What are the problems caused by domestic violence?",
                     index_name="processed_10000_data"))
    ground_truth_process_0 = ["c5e25672-812b-43a0-b558-9cf89f4f1c3d", "0d25ebac-299e-47d2-a76b-03e3cb069222",
                              "3d7ca512-e7c7-4ea2-97ea-3d4885f7583f", "49a6740c-8143-41df-b90f-63d6ddcc4804",
                              "a92c78dd-48fe-4752-9a70-dba27e606288"]

    predictions_process_1 = get_relevant_docs_ids_list(search_index(input_query="What are people fleeing from in Syria?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_1 = ["8402fd9a-10a9-406b-8a6e-57d295388522", "4caf2601-4da7-4b13-8b79-ea57a6ef7e9b",
                              "7d5db63e-efe3-4fd3-917c-7f9682b861c9", "cfccd4c7-0898-41cb-9f84-4d11267968fa",
                              "b76fbe7d-c648-49ef-9fd4-6d7759194e11", "30d58186-6f13-423d-9616-de3c6be5e8b3",
                              "eb7b1db2-53c6-4275-941a-35814e859c57", "780ece8a-7dde-44b6-b2bf-801c7d85b6da",
                              "a1ce1914-3129-4e10-b466-405639d27dee", "c6083819-8fd7-4359-b355-2fc7d3204155"]

    predictions_process_2 = get_relevant_docs_ids_list(search_index(input_query="What was discovered on Mars?",
                                                                    index_name="processed_10000_data"))
    ground_truth_process_2 = ["1eba3050-1cd2-4184-be76-bbe6f0b8f024", "5a07213b-7e81-4e0c-9dfa-758ae4c1ae3c",
                              "13ebe973-e727-47d5-8fe7-a27e05c2d578", "33ede757-4c7e-44ad-86be-c7401061af32",
                              "26a92e12-f6da-45be-afc9-4f13f699196c", "f0e7b753-2351-497a-a861-75ca339432c8"]

    predictions_process_3 = get_relevant_docs_ids_list(
        search_index(input_query="Which artists are the headliners at Apple Music Festival?",
                     index_name="articles_10000_data"))

    ground_truth_process_3 = ["0a452c76-196d-45d9-b104-09ba91570f1c", "7665c596-5e5f-4fdb-8ac8-eaa0bec7f4e9",
                              "781ec5d1-c8d4-46a5-ac94-5786c65c48fb"]

    predictions_process_4 = get_relevant_docs_ids_list(
        search_index(input_query="Which country launched its first space observatory into space?",
                     index_name="articles_10000_data"))
    ground_truth_process_4 = ["015eb5ca-1db1-4f09-adfc-58ad0ac3e456", "44611997-6d4b-4070-ba84-2cd46184b5b6"]

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
        search_index(input_query="How much does the iPad Pro cost?",
                     index_name="articles_100_data"))
    ground_truth_100_standard_0 = ["402f6983-3300-4546-82c5-6f4eb08d0ba7"]

    predictions_100_standard_1 = get_relevant_docs_ids_list(
        search_index(input_query="How many people flee to Germany?",
                     index_name="articles_100_data"))
    ground_truth_100_standard_1 = ["0d032ac6-723f-48ca-b2a8-5b5e8d55c517"]

    predictions_100_standard = [predictions_100_standard_0, predictions_100_standard_1]
    ground_truth_100_standard = [ground_truth_100_standard_0, ground_truth_100_standard_1]

    evaluation(index_name="\nArticles 100 data standard ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_standard,
               predictions_list=predictions_100_standard)

    predictions_100_process_0 = get_relevant_docs_ids_list(
        search_index(input_query="How much does the iPad Pro cost?",
                     index_name="processed_100_data"))
    ground_truth_100_process_0 = ["402f6983-3300-4546-82c5-6f4eb08d0ba7"]

    predictions_100_process_1 = get_relevant_docs_ids_list(
        search_index(input_query="How many people flee to Germany?",
                     index_name="processed_100_data"))
    ground_truth_100_process_1 = ["0d032ac6-723f-48ca-b2a8-5b5e8d55c517"]

    predictions_100_process = [predictions_100_process_0, predictions_100_process_1]
    ground_truth_100_process = [ground_truth_100_process_0, ground_truth_100_process_1]

    evaluation(index_name="\nArticles 100 data process ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_process,
               predictions_list=predictions_100_process)

    # print(search_vector(input_query="How much does the iPad Pro cost?"))
    predictions_100_vector_0 = get_relevant_docs_ids_list(
        search_vector(input_query="How much does the iPad Pro cost?"))
    ground_truth_100_vector_0 = ["402f6983-3300-4546-82c5-6f4eb08d0ba7"]

    # print(search_vector(input_query="How many people flee to Germany?"))
    predictions_100_vector_1 = get_relevant_docs_ids_list(
        search_vector(input_query="How many people flee to Germany?"))
    ground_truth_100_vector_1 = ['0d032ac6-723f-48ca-b2a8-5b5e8d55c517']

    predictions_100_vector = [predictions_100_vector_0, predictions_100_vector_1]
    ground_truth_100_vector = [ground_truth_100_vector_0, ground_truth_100_vector_1]

    evaluation(index_name="\nArticles 100 data vector ",
               num_eval_docs=2,
               ground_truth_list=ground_truth_100_vector,
               predictions_list=predictions_100_vector)


