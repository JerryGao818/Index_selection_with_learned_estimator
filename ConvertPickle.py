import json
import pickle
import os

query_list = []
not_included = ['28a.sql','28c.sql']
sqlfiles = os.listdir('./CostEstimator/data/sqlfile/')
for file in sqlfiles:
    if not file.endswith('.sql') or file in not_included:
        continue
    print(file)
    file = './CostEstimator/data/sqlfile/' + file
    sql = open(file, 'r').read()
    query_list.append(sql)



pickle.dump(query_list, open('./Entry/JOBworkload.pickle', 'wb'))

# cands = pickle.load(open('./Entry/all_cands.pickle', 'rb'))
# new_cands = []
# print(len(cands))
# for i in range(len(cands)):
#     new_cands.append('idx' + str(i))
# pickle.dump(new_cands, open('./Entry/candidates.pickle', 'wb'))

# cands = []
# count = 0
# files = os.listdir('./CostEstimator/data/OriginalData/all_index/')
# for file in files:
#     if not file.endswith('.json'):
#         continue
#     count += 1
#     file = './CostEstimator/data/OriginalData/all_index/' + file
#     indexes = json.load(open(file, 'r'))[1]
#     for index in indexes:
#         if index.startswith('idx') and index not in cands:
#             cands.append(index)
# print(cands)
# print(len(cands))
# print(count)
# pickle.dump(cands, open('./Entry/candidates.pickle', 'wb'))

