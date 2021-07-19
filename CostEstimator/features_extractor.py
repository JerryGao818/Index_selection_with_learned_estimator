#encoding: utf-8
from .PlanParser import *
import numpy as np
from .functions import get_degree

max_len = 0


def get_used_indexes(subplan, indexes):
    if 'Index Name' in subplan:
        indexes.append(subplan['Index Name'])
    if 'Plans' in subplan:
        for plan in subplan['Plans']:
            get_used_indexes(plan, indexes)


#[2,3,5,6]
def get_indexes(data):
    indexes = []
    all_info = []
    # print(data)
    imean, istd, imax, imin = normalizeIndex()
    get_used_indexes(data['Plan'], indexes)
    # print(indexes)
    for index in indexes:
        filename = '../CostEstimator/data/IndexInformation/' + index + '.json'
        if os.path.exists(filename):
            info = json.load(open(filename, 'r'))[1]
            # print(info)
            info[4] = int(info[4])
            index_natts = info[-1]
            del info[-1]
            info[2] = (info[2] - imean[0]) / (imax[0] - imin[0])#istd[k]
            info[3] = (info[3] - imean[1]) / (imax[1] - imin[1])  # istd[k]
            info[5] = (info[5] - imean[2]) / (imax[2] - imin[2])  # istd[k]
            info[6] = (info[6] - imean[3]) / (imax[3] - imin[3])  # istd[k]
            temp = np.zeros(10)
            for i in index_natts:
                if '0' <= i <= '9':
                    temp[int(i) - 1] = 1
            for i in temp:
                info.append(i)
            # print(info)
            all_info.append(info)
        else:
            print(filename, ' not existed')
    return all_info


def normalizeIndex():
    index_files = os.listdir('../CostEstimator/data/IndexInformation/')
    tuples = []
    pages = []
    attrs = []
    total_tuples = []
    for file in index_files:
        if not file.endswith('.json') or file == 'index_dic.json':
            continue
        file = '../CostEstimator/data/IndexInformation/' + file
        info = json.load(open(file, 'r'))[1]
        total_tuples.append(info[2])
        tuples.append(info[3])
        pages.append(info[5])
        attrs.append(info[6])
    static = [total_tuples, tuples, pages, attrs]
    static = np.array(static)
    # print(static)
    return (np.mean(static, axis=1), np.std(static, axis=1), np.max(static, axis=1), np.min(static, axis=1))


def generateKeywordFromTableList(table_list):
    table_set = list(set(table_list))
    res = []
    for table in table_set:
        res += [table+"."+key for key in data[table]]
    return res

def parseQuery(data,need_input,use_cost=False):
    plan = data['Plan']
    alias2table = {}
    get_alias2table(plan, alias2table)
    # print(alias2table)
    subplan, cost, cardinality = get_plan(plan,use_cost)
    # print("subplan:\n", subplan)
    seq, _, static_seq, input_tables = plan2seq(subplan, alias2table)
    # print(seq)
    seqs = PlanInSeq(seq, cost, cardinality)
    js = json.loads(class2json(seqs))
    json_seq = json2seq(js)
    if need_input:
        input_keywords = generateKeywordFromTableList(input_tables)
        input_keywords += input_tables
    else:
        input_keywords = None

    return json_seq,static_seq,input_keywords

def getLatency(file):
    f_open = open(file,'r')
    plan = json.load(f_open)['Plan']
    return plan['Actual Total Time']


def getExecutionTime(data):
    if 'Execution Time' in data:
        if data['Execution Time'] < 1800000:
            time = data['Execution Time']
        else:
            time = 1800000
    else:
        time = 1800000
    return time


def parsePath(path,splitName=False,isLatency=False,use_cost=False):
    fileNames = os.listdir(path)
    r_d = {}
    for file in fileNames:
        if not file.endswith("json"):
            continue
        key = file[:-5]
        if "mv" not in key:
            key = key.split('_')[1]
        else:
            key = key[2:]
        if splitName:
            keys = key.split('-')
            key = tuple(keys)
        file = path + "/" + file
        input_keywords = splitName is False
        if isLatency is False:
            res_1,res_2,res_3 = parseQuery(file,input_keywords,use_cost)
            r_d[key] = (res_1,res_2,res_3)
        else:
            r_d[key] = getLatency(file)
    return r_d


def build_test():
    global max_len
    # q_dict = parsePath()
    keywords = []
    files = os.listdir('./processed_data/')
    r_d = []
    count = 0
    for file in files:
        # if not file.startswith('22d'):
        #     continue
        # file = '22d-512.json'
        print(file)
        point = file.find('.')
        key = file[:point]
        file = './processed_data/' + file

        if not file.endswith('.json'):
            continue
        data = json.load(open(file, 'r'))
        # print(data)
        # file = './resource/queries/job/q_test/' + file
        res_1, res_2, res_3 = parseQuery(file, True, True)
        # print('res1: ', res_1)
        if max_len < len(res_2):
            max_len = len(res_2)
        if len(res_1['seq']) != len(res_2):
            print('file length not matched: ', file, len(res_1['seq']), len(res_2))
            raise Exception
        keywords += extractKeywords(res_1['seq'])
        keywords += res_3
        index_information = get_indexes(data)
        degree_parser = get_degree()
        degree, node_dic = degree_parser.get_degree_mat(data)
        # others = [res_1['cost'], res_1['cardinality']]
        x = (res_1['seq'], res_2, res_3, index_information, degree.tolist())
        y = getExecutionTime(data)
        r_d.append((x, y, key))
        # break
        # count += 1
        # if count > 3000:
        #     break
    return r_d, list(set(keywords))


def extractKeywords(seq):
    key_list = []
    for node in seq:
        if isinstance(node,tuple):
            continue
        for att in node:
            if att[1]==1:
                key_list.append(att[0])
    return list(set(key_list))


def build_predict_data(data):
    keywords = []
    r_d = []
    for item in data:
        res_1, res_2, res_3 = parseQuery(item, True, True)
        keywords += extractKeywords(res_1['seq'])
        keywords += res_3
        index_information = get_indexes(item)
        degree_parser = get_degree()
        degree, node_dic = degree_parser.get_degree_mat(item)
        x = (res_1['seq'], res_2, res_3, index_information, degree.tolist())
        r_d.append((x, key))
    return r_d, list(set(keywords))


if __name__ == '__main__':
    # r_l, key = buildData()
    # for i in r_l:
    #     print(i[1])
    # # print(r_l[1])
    # print(key)
    import random
    import pickle
    data, keywords = build_test()
    random.shuffle(data)
    pickle.dump(data, open('./ProgrammTemp/small_data.pickle', 'wb'))
    pickle.dump(keywords, open('./ProgrammTemp/small_keywords.pickle', 'wb'))
    # print(keywords)
    # for key in r[0][0]:
    #     print(key)
    # print(keywords)

    # data, keywords = build_test()
    # print(normalizeIndex())
    # data = json.load(open('./data/OriginalData/24/24b-66961402.json','r'))
    # get_indexes(data[0])


