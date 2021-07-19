import psycopg2
import os
import json
from DBOperator import *
from random import randint
from random import sample


database = 'imdb'
user = 'postgres'
password = 'postgres'
host = 'localhost'
port = '5432'


def get_used_indexes(subplan, indexes):
    if 'Index Name' in subplan:
        indexes.append(subplan['Index Name'])
    if 'Plans' in subplan:
        for plan in subplan['Plans']:
            get_used_indexes(plan, indexes)


def get_indexes(data):
    indexes = []
    get_used_indexes(data[0]['Plan'], indexes)
    return indexes


def create_tested_dic():
    tested_dic = {}
    dirs = os.listdir('./data_imdb/OriginalData/')
    for dir in dirs:
        # if dir != '27':
        #     continue
        if dir.startswith('.') or dir == 'normal.json':
            continue
        files = os.listdir('./data_imdb/OriginalData/' + dir + '/')
        files.sort()
        for file in files:
            if not file.endswith('json'):
                continue
            inter = file.find('-')
            name = file[:inter]
            data = json.load(open('./data_imdb/OriginalData/' + dir + '/' + file))
            used_indexes = get_indexes(data)
            if name not in tested_dic:
                tested_dic[name] = [used_indexes]
            else:
                current = tested_dic[name]
                if used_indexes not in current:
                    current.append(used_indexes)
                    tested_dic[name] = current
    json.dump(tested_dic, open('./imdb_tested_dic.json', 'w'))


def get_len(name):
    data = json.load(open('./data_imdb/CandidateIndex/' + str(name)  + 'a.json', 'r'))
    indexes = []
    for item in data:
        for index in item:
            if index not in indexes:
                indexes.append(index)
    return len(indexes)


def run_all_raw():
    # clear_index()
    sqlfiles = os.listdir('./data_imdb/sqlfile')
    print('now run raw sql...')
    errors = []
    tested = ['1.sql', '47.sql', '57.sql', '74.sql', '78.sql', '81.sql', '30.sql']
    for sqlfile in sqlfiles:
        if not sqlfile.endswith('.sql') or sqlfile in tested:
            continue
        print('now running: ', sqlfile)
        sql = 'explain (analyze, format json) ' + open('./data_imdb/sqlfile/' + sqlfile, 'r').read()
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            res1 = cur.fetchall()[0][0][0]
            print('execution time: ', res1['Execution Time'])
            used_index = get_indexes(res1)
            cur.execute(sql)
            res2 = cur.fetchall()[0][0][0]
            exe_time2 = res2['Execution Time']
            print('execution time2: ', exe_time2)
            path = './data_imdb/OriginalData/all_index/' + sqlfile.split('.')[0] + '-0.json'
            if res1['Execution Time'] < res2['Execution Time']:
                json.dump([res1, used_index], open(path, 'w'))
            else:
                json.dump([res2, used_index], open(path, 'w'))
        except Exception as e:
            print('error: ', e)
            errors.append(sqlfile)
        cur.close()
        conn.close()
        tested.append(sqlfile)
        print('current errors: ', errors, 'current tested: ', len(tested), ' ', tested)



def get_used_indexes(subplan, indexes):
    if 'Index Name' in subplan:
        indexes.append(subplan['Index Name'])
    if 'Plans' in subplan:
        for plan in subplan['Plans']:
            get_used_indexes(plan, indexes)


def get_indexes(data):
    indexes = []
    get_used_indexes(data['Plan'], indexes)
    return indexes

import pickle
created = []
def start2():
    global created
    data_collected = 0
    index_dic = json.load(open('./data_imdb/IndexInformation/index_dic.json', 'r'))
    not_included = ['28c.sql', '28a.sql']
    tested_dic = json.load(open('./imdb_tested_dic.json','r'))
    sqlfiles = os.listdir('./data_imdb/sqlfile')
    candidate = pickle.load(open('./data_imdb/CandidateIndex/IMDBcandidate3.pickle', 'rb'))
    print('candidate size:', len(candidate))
    index_count = randint(1, 20)
    current_index = sample(candidate, index_count)
    index_count = len(candidate)
    print('current count: ', index_count, ' current solution: ', current_index)
    for index in created:
        if index not in current_index:
            try:
                delete_index(index)
            except Exception as e:
                print(e)
            created.remove(index)
    for index in current_index:
        if index not in created:
            try:
                create_index([index])
            except Exception as e:
                print(e)
            created.append(index)
    for sqlfile in sqlfiles:
        if not sqlfile.endswith('.sql'):
            continue
        print('now running: ', sqlfile)
        sql0 = 'explain (format json) ' + open('./data_imdb/sqlfile/' + sqlfile, 'r').read()
        sql1 = 'explain (analyze, format json) ' + open('./data_imdb/sqlfile/' + sqlfile, 'r').read()
        point = sqlfile.find('.')
        name = sqlfile[:point]
        if name not in tested_dic:
            tested_dic[name] = []
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cur = conn.cursor()
        cur.execute(sql0)
        res = cur.fetchall()[0][0][0]
        used_index = get_indexes(res)
        flag = 0
        for index in created:
            idx_name = index_dic[index]
            if idx_name in used_index:
                flag = 1
                break
        if flag == 0:
            print('not related.')
            continue
        filename = randint(1,1000000)
        path = './data_imdb/OriginalData/all_index/' + name + '-' + str(filename) + '.json'
        while(os.path.exists(path)):
            filename = randint(1, 1000000)
            path = './data_imdb/OriginalData/all_index/' + name + '-' + str(filename) + '.json'
        print('file:', name + '-' + str(filename) + '.json')
        is_tested = tested_dic[name]
        if used_index in is_tested:
            print('file has been tested in tested_dic')
            continue
        else:
            is_tested.append(used_index)
            tested_dic[name] = is_tested
        cur.execute('set statement_timeout = 1800000;')
        try:
            cur.execute(sql1)
            res1 = cur.fetchall()[0][0][0]
            exe_time1 = res1['Execution Time']

            print('execution time1: ', exe_time1)
            cur.execute(sql1)
            res2 = cur.fetchall()[0][0][0]
            exe_time2 = res2['Execution Time']
            print('execution time2: ', exe_time2)
        except psycopg2.errors.QueryCanceled as e:
            print('sql time out. pass.')
            json.dump([res, used_index], open('./data_imdb/OriginalData/all_index/' + name + '-' + str(filename) + '.json', 'w'))
            continue
        if res1['Execution Time'] < res2['Execution Time']:
            json.dump([res1, used_index], open(path, 'w'))
        else:
            json.dump([res2, used_index], open(path, 'w'))
        data_collected += 1
        cur.close()
        conn.close()
    json.dump(tested_dic, open('imdb_tested_dic.json', 'w'))
    print('data collected: ', data_collected)

if __name__ == '__main__':
    create_tested_dic()
    clear_index()
    run_all_raw()
    for i in range(2000):
        print('*' * 30)
        print('epoch: ', i)
        start2()
    print('\n\nnow cleaning...')
    clear_index()
