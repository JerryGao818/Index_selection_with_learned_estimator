import os
from configparser import ConfigParser
from typing import List
import psycopg2 as pg
import pandas as pd
import time
import json
import re
import sys
sys.path.append('..')
import torch
import pickle
from CostEstimator.features_extractor import *
# from CostEstimator.features_extractor import build_predict_data
from CostEstimator.CostEstimator import WideDeep

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device = torch.device('cuda:0')


def findkey(plan):
    # print('check subplan')
    if 'Index Name' in plan:
        if plan['Index Name'] == 'web_site_pkey':
            return True
    if 'Plans' in plan:
        for subplan in plan['Plans']:
            if findkey(subplan):
                return True
    return False

class PGHypo:
    def __init__(self):
        config_raw = ConfigParser()
        config_raw.read(os.path.abspath('..') + '/configure.ini')
        defaults = config_raw.defaults()
        self.host = defaults.get('pg_ip')
        self.port = defaults.get('pg_port')
        self.user = defaults.get('pg_user')
        self.password = defaults.get('pg_password')
        self.database = defaults.get('pg_database')
        self.conn = pg.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                                     port=self.port)

    def close(self):
        self.conn.close()

    def show_all(self):
        sql = 'select * from hypopg_list_indexes()'
        cur = self.conn.cursor()
        cur.execute(sql)
        res = cur.fetchall()
        # print(res)
        return res

    def execute_create_hypo(self, index):
        real_index = json.load(open('../CostEstimator/data/IndexInformation/' + index + '.json'))[0]
        # print(real_index)
        schema = real_index.split("#")
        sql = "SELECT indexrelid FROM hypopg_create_index('CREATE INDEX ON " + schema[0] + "(" + schema[1] + ")') ;"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        new_oid = rows[0][0]
        # print(new_oid)
        index_name_dic = json.load(open('../namedic.json', 'r'))
        current_indexes = self.show_all()
        # print(current_indexes)
        # print(index_name_dic)
        for item in current_indexes:
            if item[0] == new_oid:
                index_name_dic[item[1]] = index
        # print(index_name_dic)
        json.dump(index_name_dic, open('../namedic.json', 'w'))
        return int(rows[0][0])

    def execute_delete_hypo(self, oid):
        sql = "select * from hypopg_drop_index(" + str(oid) + ");"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        flag = str(rows[0][0])
        # if flag == "t":
        index_name_dic = json.load(open('../namedic.json', 'r'))
        for key in index_name_dic:
            # print(key)
            pattern = re.compile('<([0-9]*)>')
            res = pattern.search(key)
            if str(oid) == res.group(1):
                index_name_dic.pop(key)
                json.dump(index_name_dic, open('../namedic.json', 'w'))
                break
        if flag == "True":
            return True
        return False

    def transfer_index_name(self, plan):
        if "Index Name" in plan:
            name = plan["Index Name"]
            index_name_dic = json.load(open('../namedic.json', 'r'))
            if name in index_name_dic:
                plan["Index Name"] = index_name_dic[name]
                # print(name,' -> ',plan['Index Name'])
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self.transfer_index_name(subplan)

    def get_queries_cost(self, query_list):
        # print('Now estimate cost using deep learning model...')
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        data = []
        for i, query in enumerate(query_list):
            query = "explain (format json) " + query
            # print(query)
            cur.execute(query)
            rows = cur.fetchall()
            rows = rows[0][0][0]
            # print(rows)
            self.transfer_index_name(rows['Plan'])
            if findkey(rows['Plan']):
                print(query)
                print(rows)
                print(i)
            data.append(rows)
        # print('data collected')
        # print(data[0])
        current_data, keywords = build_predict_data(data)
        # for i in range(len(current_data)):
        #     for j in range(len(current_data[i][2]))
        data_statistic = np.load('../CostEstimator/data_statistic.npy')
        mean1 = data_statistic[0]
        std1 = data_statistic[1]
        # print(len(current_data))
        for i in range(len(current_data)):
            for j in range(len(current_data[i][0][1])):
                current_data[i][0][1][j][0] = (current_data[i][0][1][j][0] - mean1[0]) / std1[0]  # (max1[0] - min1[0])
                current_data[i][0][1][j][1] = (current_data[i][0][1][j][1] - mean1[1]) / std1[1]  # (max1[1] - min1[1])
                current_data[i][0][1][j][2] = (current_data[i][0][1][j][2] - mean1[2]) / std1[2]  # (max1[2] - min1[2])
        keywords = pickle.load(open('../CostEstimator/ProgramTemp/keywords.pickle', 'rb'))
        args = {}
        args['keywords'] = keywords
        args['keyword_embedding_size'] = 32
        args['char_embedding_size'] = 64
        args['node_auxiliary_size'] = 3
        args['first_hidden_size'] = 64
        args['drop_rate'] = 0.2
        args['other_size'] = len(current_data[0][0][-1])
        args['key2index'] = {word: index for index, word in enumerate(keywords)}
        args['index2key'] = {value: key for key, value in args['key2index'].items()}
        args['index_info_size'] = 17
        args['q_max_len'] = 50
        args['graph_learner_size'] = 77
        args['gcn_out_size'] = 32
        args['max_index_count'] = 13
        model = GCNCost(args)
        model = model.to(device)
        checkpoint = torch.load('../CostEstimator/ProgramTemp/GCN_Attention_Estimator_0.001_1e-05-32-64-3-64-0.2.model')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        test_data = [item[0] for item in current_data]
        # test_data = [item[0] for item in data]
        res = model(test_data)
        res = res.cpu()
        _cost_list = res.squeeze().detach().numpy()
        label_statistic = np.load('../CostEstimator/label_statistic.npy')
        # print(label_statistic)
        l_mean = label_statistic[0]
        l_std = label_statistic[1]
        cost_list = _cost_list * l_std + l_mean
        cost_list = np.exp(cost_list)
        # print(res, np.exp(res.squeeze().detach().numpy()))
        # print('estimated total cost: ', cost_list.sum())
        return cost_list.tolist()

    def get_storage_cost(self, oid_list):
        costs = list()
        cur = self.conn.cursor()
        for i, oid in enumerate(oid_list):
            if oid == 0:
                continue
            sql = "select * from hypopg_relation_size(" + str(oid) +");"
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_long = int(cost_info)
            costs.append(cost_long)
            # print(cost_long)
        return costs

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def delete_indexes(self):
        sql = 'select * from hypopg_reset();'
        self.execute_sql(sql)
        empty = {}
        json.dump(empty, open('../namedic.json', 'w'))

    def get_sel(self, table_name, condition):
        cur = self.conn.cursor()
        totalQuery = "select * from " + table_name + ";"
        cur.execute("EXPLAIN " + totalQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from " + table_name + " Where " + condition + ";"
        # print(resQuery)
        cur.execute("EXPLAIN  " + resQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows/total_rows

    def get_rel_cost(self, query_list):
        print("real")
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            print('\r now execute {0}th query... '.format(i), end='')
            cur.execute(query)
            # _start = time.time()
            query = "explain (analyse, format json) " + query
            # cur.execute(query)
            # data1 = cur.fetchall()[0][0][0]
            cur.execute(query)
            data = cur.fetchall()[0][0][0]
            # _end = time.time()
            # print(data)
            time = data['Execution Time']
            cost_list.append(time)
            print('done. execution time: ', time)
        cost = np.array(cost_list).sum()
        return cost

    def create_indexes(self, indexes):
        i = 0
        for index in indexes:
            schema = index.split("#")
            sql = 'CREATE INDEX START_X_IDx' + str(i) + ' ON ' + schema[0] + "(" + schema[1] + ');'
            print(sql)
            self.execute_sql(sql)
            i += 1

    def delete_t_indexes(self):
        sql = "SELECT relname from pg_class where relkind = 'i' and relname like 'start_x_idx%';"
        print(sql)
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        indexes = []
        for row in rows:
            indexes.append(row[0])
        print(indexes)
        for index in indexes:
            sql = 'drop index ' + index + ';'
            print(sql)
            self.execute_sql(sql)

    def get_tables(self, schema):
        tables_sql = 'select tablename from pg_tables where schemaname=\''+schema+'\';'
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])
        return table_names

    def get_attributes(self, table_name, schema):
        attrs_sql = 'select column_name, data_type from information_schema.columns where table_schema=\''+schema+'\' and table_name=\''+table_name+'\''
        cur = self.conn.cursor()
        cur.execute(attrs_sql)
        rows = cur.fetchall()
        attrs = list()
        for i, attr in enumerate(rows):
            info = str(attr[0]) + "#" + str(attr[1])
            attrs.append(info)
        return attrs

    def clear_index(self):
        delete_count = 0
        cur = self.conn.cursor()
        search_index = '''select relname from pg_class where relkind='i' and relname !~ '^(pg_|sql_)';'''
        cur.execute(search_index)
        current_index = cur.fetchall()
        print('current indexes are: ', current_index)
        print('below will be removed: ')
        for index in current_index:
            if not index[0].endswith('_pkey'):
                print(index[0])
                delete_count = delete_count + 1
                delete_statement = 'drop index ' + index[0] + ';'
                cur.execute(delete_statement)
                self.conn.commit()
        print(delete_count, ' indexes removed.')

if __name__ == '__main__':
    pg = PGHypo()
    # pg.clear_index()
    # import pickle
    # pg.execute_create_hypo('idx1')
    query_list = pickle.load(open('../Entry/JOBworkload.pickle', 'rb'))
    _cost = pg.get_queries_cost(query_list)
    print(np.array(_cost).sum())
    cost = pg.get_rel_cost(query_list)
    print(cost)
