from preprocess import *
from utils import *
from Gensim_Model import Gensim_Model
from CustomEnumerators import TopicModelingAlgorithm, CoherenceType
from python_to_html import create_html
import os
import numpy as np

# Return a dictionary with each endpoint in every topic it had an score over 0 [id of endpoint, topic it belongs, % of topic]
def endpoints_for_topic(mod, tpc):
    dataset = create_table_ids_topics(mod)
    d_end = dataset_topics_over_zero(dataset)
    dict_end = {i: (d_end[i][j][0], d_end[i][j][1]) for i in range(len(d_end)) for j in range(len(d_end[i])) if d_end[i][j][0] == tpc}
    ordered_d = sorted(dict_end.items(), key=lambda x:x[1][1], reverse=True)
    return ordered_d

# Return a dictionary with the endpoints assigned to the topics with their highest score [id of endpoint, topic it belongs, % of topic]
def best_endpoints_for_topic(mod, tpc):
    dataset = create_table_ids_topics(mod)
    d_oz = dataset_topics_over_zero(dataset)
    dict_best = {}
    for i in range(len(d_oz)):
        best_score = -1
        best_tpc = None
        for j in range(len(d_oz[i])):
            if(d_oz[i][j][1] > best_score):
                best_score = d_oz[i][j][1]
                best_tpc = d_oz[i][j][0]
        if(best_tpc == tpc):
            dict_best[i] = (best_tpc, best_score)
    ordered_d = sorted(dict_best.items(), key=lambda x:x[1][1], reverse=True)
    return ordered_d

def gen_csv_to_topics(mod, bests, id_info):
    csv_vals = []
    for i in range(mod.cur_model_topic_quantity()):
        for j in range(5):
            csv_vals.append(i)
            csv_vals.append(', '.join([elem[0] for elem in mod.cur_model_topic_words(i, 5)]))
            endp_id = bests[i][j][0]
            endp_score = round(bests[i][j][1][1], 3)
            info = id_info[endp_id]
            csv_vals.append(info[0] + info[1])
            csv_vals.append(endp_score)

    return csv_vals

def gen_csv_to_statistics(mod, bests, id_info):
    csv_vals = []
    tpcs = {}
    top_topics = mod.cur_model_top_topics_by_coh()
    top_topics = {', '.join([word[1] for word in top_topics[i][0]]): top_topics[i][1] for i in range(len(top_topics))}

    #print(top_topics)

    for i in range(mod.cur_model_topic_quantity()):
        words_tpc = [elem[0] for elem in mod.cur_model_topic_words(i, 5)]
        weights_w_tpc = [elem[1] for elem in mod.cur_model_topic_words(i, 5)]
        tpc = ', '.join(words_tpc)
        tpcs[i] = top_topics[tpc]
        best_i = [bests[i][j][1][1] for j in range(len(bests[i]))]
        avg_score, std_score = np.average(best_i), np.std(best_i)
        avg_weights, std_weights = np.average(weights_w_tpc), np.std(weights_w_tpc)
        qtt_endpoints = len(bests[i])
        csv_vals.append(i)
        csv_vals.append(round(tpcs[i], 3))
        csv_vals.append(round(avg_score, 3))
        csv_vals.append(round(std_score, 3))
        csv_vals.append(round(avg_weights, 3))
        csv_vals.append(round(std_weights, 3))
        csv_vals.append(round(qtt_endpoints, 3))

    return csv_vals

def create_table_ids_topics(gen_mod):
    rec = gen_mod.endpoint_in_topics_dataset()
    df = pd.DataFrame(rec, columns=['id_sentence', 'topic_num', 'score'])
    sentSubsetGroup = df.groupby(['id_sentence'])
    sentSubsetGroup = sorted(sentSubsetGroup, key=lambda x: len(x[1]), reverse=True)

    return sentSubsetGroup

def evaluate_several_models(component, algorithms = [TopicModelingAlgorithm.LDA, TopicModelingAlgorithm.LSA, TopicModelingAlgorithm.NMF], qtt_topics = range(2,21), result_folder=None):
    openAPI_dataset_directory = './APIs/'
    data_file = './data.txt'
    data_file_exists = os.path.isfile(data_file)

    if(not data_file_exists):
        #The following function will read all the information in openAPI_dataset_directory
        #and put it in a file (data_file), for faster loading
        data_list = generate_list(data_file, openAPI_dataset_directory)
    else:
        #If the file with the APIs info already exist, we can load it
        data_list = load_list(data_file)

    print('OpenAPI directory data loaded')

    #The following method filters the endpoints that has both summary and description
    #Then it returns the [component] (summary or description) of each endpoint, along
    #with a series of information for each of these endpoints, by id (id_info)
    data_list, id_info = get_component_if_both_and_info(data_list, component)

    #Here we execute all the pre-processing of the text in each endpoint, also deleting the
    #endpoints that are empty (they might have lost all the words in the pre-processing,
    #because of stop-words, lemmatization and etc)
    data_lemmatized, id_info = execute_preprocessing_and_update_info(data_list, id_info, None)
    print('Data pre-processed')

    mod = Gensim_Model(data_lemmatized)
    mod.evaluate_several_models(algorithms, qtt_topics, out_folder=result_folder, out_file='results_' + component)
    mod.set_best_model()

    return mod

def main():
    openAPI_dataset_directory = './APIs/'
    data_file = './data.txt'
    data_file_exists = os.path.isfile(data_file)

    if(not data_file_exists):
        #The following function will read all the information in openAPI_dataset_directory
        #and put it in a file (data_file), for faster loading
        data_list = generate_list(data_file, openAPI_dataset_directory)
    else:
        #If the file with the APIs info already exist, we can load it
        data_list = load_list(data_file)

    print('OpenAPI directory data loaded')

    #The following method filters the endpoints that has both summary and description
    #Then it returns the [component] (summary or description) of each endpoint, along
    #with a series of information for each of these endpoints, by id (id_info)
    component = 'description'
    data_list, id_info = get_component_if_both_and_info(data_list, component)

    #Here we execute all the pre-processing of the text in each endpoint, also deleting the
    #endpoints that are empty (they might have lost all the words in the pre-processing,
    #because of stop-words, lemmatization and etc)
    data_lemmatized, id_info = execute_preprocessing_and_update_info(data_list, id_info, None)
    print('Data pre-processed')

    # We create the model, based on the coherence that we have got in the method that
    #tests several algorithms and topic quantity
    mod = Gensim_Model(data_lemmatized)
    mod.set_model(TopicModelingAlgorithm.NMF, 12)
    print('Model set')

    #Once the model is completed, we will create a list of lists with the endpoints
    #assigned to the topic their score is highest, the index of the
    #list represents the topic and inside it, we have each endpoint. So... bests[0] show
    #all the endpoints that are better represented by the topic 0
    bests = [best_endpoints_for_topic(mod, i) for i in range(mod.cur_model_topic_quantity())]
    print('Endpoints assigned to topics')

    #2 csvs are needed for the generation of the html file with the results.
    #the first method generates the csv with the information on the topics modelled and endpoints
    #the second method generates the csv with statistical information on the topics and endpoint distribution
    csv_topics = gen_csv_to_topics(mod, bests, id_info)
    csv_statistics = gen_csv_to_statistics(mod, bests, id_info)
    print('CSV files created')

    # This will create the html file with the results. It's named "index.html" by defaul so
    # we can access it via github.io
    create_html(csv_topics, csv_statistics, 'index.html')
    print('HTML file created. Done.')

if __name__ == "__main__":
    main()
