import json
import os


def get_used_indexes(subplan, indexes):
    if 'Index Name' in subplan:
        indexes.append(subplan['Index Name'])
    if 'Plans' in subplan:
        for plan in subplan['Plans']:
            get_used_indexes(plan, indexes)


def get_indexes(file):
    indexes = []
    name = file.find('-')
    name = file[:name-1]
    if os.path.exists('./data/OriginalData/' + name + '/' + file):
        data = json.load(open('./data/OriginalData/' + name + '/' + file, 'r'))
    else:
        data = json.load(open('./data/OriginalData/all_index/' + file, 'r'))
    # print(data)
    get_used_indexes(data[0]['Plan'], indexes)
    return indexes


def getExecutionTime(file):
    name = file.find('-')
    name = file[:name-1]
    if os.path.exists('./data/OriginalData/' + name + '/' + file):
        data = json.load(open('./data/OriginalData/' + name + '/' + file, 'r'))
    else:
        data = json.load(open('./data/OriginalData/all_index/' + file, 'r'))
    if 'Execution Time' in data:
        if data['Execution Time'] < 5400000:
            time = data['Execution Time']
        else:
            time = 5400000
    else:
        time = 5400000
    return time

def process_data():
    count = 0
    files = os.listdir('./data/OriginalData/all_index/')
    files.sort()
    name = files[0][:files[0].find('-')]
    for file in files:
        if file.startswith('.'):
            continue
        count += 1
        data = json.load(open('./data/OriginalData/all_index/' + file, 'r'))[0]
        json.dump(data, open('./processed_data/' + file, 'w'))
        # if count >= 100:
        #     break
    print('%d files processed.' % count)
