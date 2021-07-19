import pickle
import json

def find_overlap():
    can = pickle.load(open('./JOBcandidate.pickle', 'rb'))
    print(can)
    print(len(can))
    candi = []
    for i in can:
        candi.append(json.load(open('../CostEstimator/data/IndexInformation/' + i + '.json', 'r'))[0])
    print(can)
    print(candi)
    overlap = {}
    for id in range(len(candi)):
        schema = candi[id].split("#")
        attrs = schema[1].split(',')
        subindex = schema[0] + '#'
        subid = []
        if len(attrs) == 1:
            overlap[id] = subid
            continue
        i = 0
        while i < len(attrs):
            subindex += attrs[i]
            if subindex in candi and subindex != candi[id]:
                subid.append(candi.index(subindex))
            if i < len(attrs) - 1:
                subindex += ','
            i += 1
        overlap[id] = subid

    for id in overlap:
        print(candi[id], ': ', end='')
        subid = overlap[id]
        for i in subid:
            print(candi[i], end='; ')
        print(' ')

    print(overlap)
    pickle.dump(overlap, open('../Entry/indexOverlap.pickle', 'wb'))


