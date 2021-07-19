import numpy as np
import pickle
import os
import sys
from .findOverlap import find_overlap
sys.path.append('..')

import Model.Model3DQNFixStorage as model2
import Utility.PostgreSQLwithEstimator as pg
from Enviornment.Env3DQNFixStorage import get_index_storage


def One_Run_DQN(conf, __x, is_dnn, is_ps, is_double, a):
    conf['NAME'] = + str(__x)
    print('=====load workload=====')
    wf = open('JOBworkload.pickle', 'rb')
    workload = pickle.load(wf)
    print('=====load candidate =====')
    cf = open('JOBcandidate.pickle', 'rb')
    index_candidates = pickle.load(cf)
    print('storage limit: ', __x)
    agent = model2.DQN(workload, index_candidates, 'hypo', conf)
    _indexes, storages = agent.train(False, __x)
    indexes = []
    for _i, _idx in enumerate(_indexes):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])
    return indexes


def eval_real_cost(indexes):
    cost = np.array(get_index_storage(indexes)).sum()
    cost = cost/(1024*1024)
    print('index count:', len(indexes),' storage cost: ', cost)
    pg_client = pg.PGHypo()
    pg_client.clear_index()
    pg_client.create_indexes(indexes)
    wf = open('IMDBworkload.pickle', 'rb')
    workload = pickle.load(wf)
    cost = pg_client.get_rel_cost(workload)
    print('cost: ', cost)
    pg_client.clear_index()


pickle.dump({}, open('./namedic.json', 'wb'))


conf = {'LR': 0.01, 'EPISILO': 0.9, 'Q_ITERATION': 9, 'U_ITERATION': 3, 'BATCH_SIZE': 16, 'GAMMA': 0.9,
        'EPISODES': 1600, 'LEARNING_START': 400, 'MEMORY_CAPACITY': 800}

def entry(constraint):
    print(One_Run_DQN(conf, constraint, False, False, False, 0))

if not os.path.exists('../Entry/indexOverlap.pickle'):
    find_overlap()


entry(64)





