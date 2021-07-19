import numpy as np
import sys
sys.path.append('..')
from Utility import PostgreSQLwithEstimator as pg
import math
from typing import List
import sys
import json
import pickle



class Env:
    def __init__(self, workload, candidates, mode):
        self.workload = workload
        self.candidates = candidates
        # create real/hypothetical index
        self.mode = mode
        self.pg_client1 = pg.PGHypo()
        self.pg_client2 = pg.PGHypo()
        # if use frequencies, input it
        # self._frequencies = []
        # self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()
        # self.frequencies = self._frequencies / np.array(self._frequencies).sum()
        #end

        self._frequencies = np.ones(len(workload))
        self.frequencies = self._frequencies

        # index overlap
        self.index_overlap = pickle.load(open('../Entry/indexOverlap.pickle', 'rb'))

        # state info
        self.init_cost = np.array(self.pg_client1.get_queries_cost(workload)) * self.frequencies
        self.init_cost_sum = self.init_cost.sum()
        self.init_state = np.append(np.zeros((len(candidates),), dtype=np.float),np.array([0, 0]))
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # utility info
        self.index_oids = np.zeros((len(candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_count = 0
        self.currenct_index = np.zeros((len(candidates),), dtype=np.float)
        self.actual_currenct_index = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(candidates),), dtype=np.float)

        # monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.storage_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_index = np.zeros((len(candidates),), dtype=np.float)

        self.current_storage_sum = 0
        self.last_reward = 0
        self.last_perf_gain = sys.maxsize
        self.max_size = 0
        self.imp_count = 0

    def step(self, action):
        # action = action[0]
        if self.currenct_index[action] != 0.0:
            return self.last_state, self.last_reward, False

        _oid = self.pg_client1.execute_create_hypo(self.candidates[action])

        overlap_indexes = self.index_overlap[action]
        for overlap_index in overlap_indexes:
            self.currenct_index[overlap_index] = 1.0
            self.actual_currenct_index[overlap_index] = 0.0
            if self.index_oids[overlap_index] > 0:
                self.current_index_count -= 1
                f = self.pg_client1.execute_delete_hypo(self.index_oids[overlap_index])
                if not f:
                    raise Exception
                self.current_storage_sum -= self.current_index_storage[overlap_index]
                print('index to create:', self.candidates[action], 'delete index:', self.candidates[overlap_index], \
                      'current storage cost:', self.current_storage_sum / 134815744)
                self.index_oids[overlap_index] = 0
                self.current_index_storage[overlap_index] = 0

        storage_cost = get_index_storage([self.candidates[action]])[0]
        self.current_storage_sum += storage_cost
        current_cost_info = np.array(self.pg_client1.get_queries_cost(self.workload)) * self.frequencies
        current_cost_sum = current_cost_info.sum()
        print('create index:', self.candidates[action], ' current storage cost:', self.current_storage_sum / 134815744, 'current sum: ', current_cost_sum, end='')
        deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        if deltac0 < 0:
            reward = -1.5
        else:
            reward = deltac0


        if self.current_storage_sum >= self.max_size:
            print('  storage exceeded')
            self.cost_trace_overall.append(self.last_cost_sum)
            self.index_trace_overall.append(self.actual_currenct_index)
            self.storage_trace_overall.append(self.current_index_storage)
            return self.last_state, self.last_reward, True
        print('  reward:', reward)
        self.index_oids[action] = _oid
        self.currenct_index[action] = 1.0
        self.actual_currenct_index[action] = 1.0
        self.current_index_storage[action] = storage_cost
        self.current_index_count += 1
        self.last_cost = current_cost_info
        self.last_cost_sum = current_cost_sum
        self.last_state = np.append(self.currenct_index, np.array([self.current_storage_sum / 134815744, self.max_size / 134815744]))
        self.last_reward = reward
        # print('current candidate:', self.actual_currenct_index)
        # print(self.last_state)
        return self.last_state, reward, False


    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        self.index_oids = np.zeros((len(self.candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_count = 0
        self.current_min_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.currenct_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.actual_currenct_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(self.candidates),), dtype=np.float)
        self.pg_client1.delete_indexes()
        self.cost_trace_overall.append(self.last_cost_sum)
        self.last_reward = 0
        self.current_storage_sum = 0
        self.last_perf_gain = sys.maxsize
        self.imp_count = 0
        self.pg_client2.delete_indexes()
        return self.last_state

def get_index_storage(indexes):
    costs = []
    for index in indexes:
        index_data = json.load(open('../CostEstimator/data/IndexInformation/' + index + '.json','r'))[1]
        costs.append(index_data[5] * 8192)
    return costs
