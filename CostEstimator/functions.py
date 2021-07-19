import json
import numpy as np
import os

#DFS algorithm to get adjacent matrix
class get_degree:
    def __init__(self):
        self.node_num = 1
        self.node_dic = {}
        self.degree = np.zeros((50, 50))   #max node number


    def fill_mat(self, id, subplan):
        if 'Plans' in subplan:
            for plan in subplan['Plans']:
                current = self.node_num
                self.node_num += 1
                i = 0
                name = plan['Node Type'] + str(i)
                while name in self.node_dic:
                    i += 1
                    name = plan['Node Type'] + str(i)
                self.node_dic[name] = current
                self.degree[id][current] = 1
                self.degree[current][id] = 1
                if 'Plans' in plan:
                    self.fill_mat(current, plan)



    def get_degree_mat(self, data):
        # node_dic = {}
        self.node_dic[data['Plan']['Node Type'] + '0'] = 0
        self.fill_mat(0, data['Plan'])
        return self.degree, self.node_dic


def get_used_indexes(subplan, indexes):
    if 'Index Name' in subplan:
        indexes.append(subplan['Index Name'])
    if 'Plans' in subplan:
        for plan in subplan['Plans']:
            get_used_indexes(plan, indexes)


def get_indexes(data):
    indexes = []
    all_info = []
    print(data)
    get_used_indexes(data[0]['Plan'], indexes)
    for index in indexes:
        filename = './data/IndexInformation/' + index + '.json'
        if os.path.exists(filename):
            info = json.load(open(filename, 'r'))
            print(info)
            info[2] = int(info[2])
            temp = np.zeros(10, dtype=np.int64)
            for i in info[-1]:
                if '0' <= i <= '9':
                    temp[int(i) - 1] = 1
            del info[-1]
            for i in temp:
                info.append(i)
            print(info)
            all_info.append(info)
        else:
            print('not existed')
    return all_info


if __name__ == '__main__':
    # file = './resource/queries/job/q_test/' + '24a-118307044.json'
    files = os.listdir('./resource/queries/job/q_test/')
    max = 0
    for file in files:
        file = './resource/queries/job/q_test/' + file
        data = json.load(open(file, 'r'))
        test = get_degree()
        deg, node_dic = test.get_degree_mat(data)
        deg = deg.tolist()
        deg2 = json.load(open('./degtest.json', 'r'))
        # print(deg == deg2)
        # for i in deg:
        #     print(i)
        if max < len(node_dic):
            max = len(node_dic)
        print(len(node_dic))
    print('max:', max)
    print(len(files))
    # file = './data/OriginalData/29/29a-524288.json'
    # data = json.load(open(file, 'r'))
    # info = get_indexes(data)
    # print(info)


