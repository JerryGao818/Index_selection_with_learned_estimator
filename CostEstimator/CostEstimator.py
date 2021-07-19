#encoding: utf-8
import numpy as np
import torch
import pandas as pd
import json
import os
import torch.nn as  nn
import random
import math
import multiprocessing as mp
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2'
mp.set_start_method('spawn')
device = torch.device('cuda:0')

from features_extractor import *



def str2int(strs):
    return list(filter(lambda x:x<128,[ord(ch)<128 for ch in strs]))


def mae_loss():
    return torch.nn.SmoothL1Loss()


def mse_loss():
    return torch.nn.MSELoss()


class MAPELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mae = torch.nn.SmoothL1Loss(reduce=False)
        self.eps = eps

    def forward(self, input, target):
        loss = 100. * torch.mean(self.mae(input, target) / (torch.abs(target) + self.eps))
        return loss


def mape_loss():
    return MAPELoss()


def batch_normalize(graph_data):
    res = []
    for item in graph_data:
        res.append(process_graph(item))
    return torch.stack(res)


def process_graph(A, symmetric = True):
    # A = A+I
    A = A + torch.eye(A.size(0)).to(device)
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)


class GCNCost(nn.Module):
    def __init__(self,args):
        super(GCNCost,self).__init__()
        self.keywords = args['keywords']
        self.keyword_embedding_size = args['keyword_embedding_size']
        self.char_embedding_size = args['char_embedding_size']
        self.node_auxiliary_size = args['node_auxiliary_size']
        self.first_hidden_size = args['first_hidden_size']
        self.drop_rate = args['drop_rate']
        self.other_size = args['other_size']
        self.key2index = args['key2index']
        self.index2key = args['index2key']
        self.index_info_size = args['index_info_size']
        self.max_index_count = args['max_index_count']
        self.q_max_len = args['q_max_len']
        self.graph_learner_size = args['graph_learner_size']
        self.gcn_out_size = args['gcn_out_size']

        self.gcn1 = nn.Linear(self.first_hidden_size + self.node_auxiliary_size, 32)
        self.gcn2 = nn.Linear(32, self.gcn_out_size)
        self.reset_parameters(self.gcn1)
        self.reset_parameters(self.gcn2)
        self.act = nn.ReLU()

        self.char_embedding = nn.Embedding(128, self.char_embedding_size)
        # encoding string
        self.reset_parameters(self.char_embedding)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(4),
            # nn.Dropout2d(self.drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(1),
            # nn.Dropout2d(self.drop_rate),
            nn.ReLU(),
        )
        self.reset_parameters(self.cnn)




        self.keyword_embedding  = nn.Embedding(len(self.keywords)+1,self.keyword_embedding_size)
        self.reset_parameters(self.keyword_embedding)
        self.keyword_aline = nn.Linear(self.keyword_embedding_size,
                                     self.char_embedding_size)
        self.reset_parameters(self.keyword_aline)


        self.graph_learner = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.reset_parameters(self.graph_learner)

        self.index_wise_attention_cnn = nn.Sequential(
            nn.Conv2d(1,1,(1,3),(1,3)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1,1,(1,3),(1,3)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, (1, 3), (1, 3)),
            nn.MaxPool2d((1,2)),
            nn.ReLU()
        )
        self.reset_parameters(self.index_wise_attention_cnn)

        self.index_wise_attention_Linear = nn.Sequential(
            nn.Linear(self.max_index_count, 5),
            nn.ReLU(),
            nn.Linear(5, self.max_index_count),
            nn.ReLU()
        )
        self.reset_parameters(self.index_wise_attention_Linear)

        self.element_wise_attention_cnn = nn.Sequential(
            nn.Conv2d(1,1,(3,1),(2,1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, (3, 1), (2,1)),
            nn.MaxPool2d((2,1)),
            nn.ReLU()
        )
        self.reset_parameters(self.element_wise_attention_cnn)

        self.element_wise_attention_Linear = nn.Sequential(
            nn.Linear(self.keyword_embedding_size * 2 + self.index_info_size - 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.keyword_embedding_size * 2 + self.index_info_size - 2),
            nn.ReLU()
        )
        self.reset_parameters(self.element_wise_attention_Linear)

        self.index_learner = nn.Sequential(
            nn.Linear(self.keyword_embedding_size * 2 + self.index_info_size - 2, 160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
        )
        self.reset_parameters(self.index_learner)


        # encoding input schema
        self.lstm1 = nn.LSTM(input_size=self.char_embedding_size,num_layers=1,hidden_size=self.first_hidden_size,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.first_hidden_size + self.node_auxiliary_size, num_layers=1,
                             hidden_size=128, batch_first=True)
        self.reset_parameters(self.lstm1)


        self.resnet1= nn.Sequential(
            nn.Linear(self.keyword_embedding_size+ self.graph_learner_size + self.max_index_count * 20, 320),
            nn.Dropout(self.drop_rate),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.Dropout(self.drop_rate),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.Dropout(self.drop_rate),
            nn.ReLU(),
            nn.Linear(320,self.keyword_embedding_size+self.graph_learner_size + self.max_index_count * 20),
            nn.Dropout(self.drop_rate),
            nn.ReLU()
        )
        self.reset_parameters(self.resnet1)





        self.regressor = nn.Sequential(
            nn.Linear(self.keyword_embedding_size+self.graph_learner_size + self.max_index_count * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )
        self.reset_parameters(self.regressor)


    def reset_parameters(self,module):
        for m in module.modules():
            if not isinstance(m, nn.Sequential) and not isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(3))
                    if hasattr(m, 'bias') and  m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)

    def encoding_index(self, input):
        vector = []
        for index in input:
            temp = []
            for item in index:
                if isinstance(item, str):
                    content = torch.LongTensor([self.key2index[item]])
                    content = content.to(device)
                    temp.append(self.keyword_embedding(content))
            temp = torch.cat(temp, dim=1).squeeze()
            vector.append(torch.cat((temp, torch.tensor(index[2:]).to(device))))
        res = torch.stack(vector)
        return res


    def encoding_seq(self, input):
        vector = []
        for node in input:
            temp = []
            if isinstance(node,tuple):
                if node[0] in self.key2index:
                    content = torch.LongTensor([self.key2index[node[0]]])
                    content = content.to(device)
                    temp.append(self.keyword_aline(self.keyword_embedding(content)).squeeze())
                else:
                    temp.append(self.keyword_aline(torch.zeros(self.keyword_embedding_size).to(device)).squeeze())
            else:
                for att in node:
                    if att[1] == 1:
                        if att[0] in self.key2index:
                            content = torch.LongTensor([self.key2index[att[0]]])
                            content = content.to(device)
                            v = self.keyword_aline(self.keyword_embedding(content)).squeeze()
                        else:
                            v = self.keyword_aline(torch.zeros(self.keyword_embedding_size).to(device)).squeeze()
                    else:
                        content = torch.LongTensor(str2int(att[0]))
                        content = content.to(device)
                        v = torch.mean(self.cnn(self.char_embedding(content).unsqueeze(0).unsqueeze(1)).reshape(len(att[0]),-1),dim=0).to(device)
                    temp.append(v)
            vector.append(torch.stack(temp))



        max_length = np.max([len(v) for v in vector])
        index = [len(v) for v in vector]
        vector = torch.stack(
            [torch.cat((v.to(device), torch.FloatTensor([[0] * self.char_embedding_size] * (max_length - len(v))).to(device))) for v in vector]).to(device)
        out, _ = self.lstm1(vector, None)
        length_mask = torch.zeros(len(index), max_length).scatter_(1, torch.LongTensor(index).unsqueeze(1) - 1,
                                                                   1).bool().reshape(len(index), max_length, 1).repeat(1,
                                                                                                                      1,
                                                                                                                     self.first_hidden_size).to(device)
        real_out = torch.masked_select(out, length_mask).view(len(index), -1)
        return real_out


    def encoding_input(self,input):
        content = torch.LongTensor([self.key2index[key] for key in input])
        content = content.to(device)
        vector = self.keyword_embedding(content)
        return torch.mean(vector,dim=0)


    def forward(self, x):
        batch_size = len(x)
        q_json_seq, q_static_seq, q_input_seq, index_info, degree_mat  = \
            [data[0] for data in x],[data[1] for data in x],[data[2] for data in x],[data[3] for data in x],[data[4] for data in x]
        # for q
        # q_static_seq = torch.FloatTensor(q_static_seq)
        # q_static_seq = q_static_seq.cuda()
        q_js_encoding = [torch.cat((self.encoding_seq(q_json_seq[i]),torch.FloatTensor(q_static_seq[i]).to(device)),dim=1) for i in range(batch_size)]
        # print(q_static_seq)
        # print('encode q: ', q_js_encoding[0][0])

        q_max_length = self.q_max_len
        q_js_encoding = torch.stack(
            [torch.cat((v, torch.FloatTensor([[0] * (self.first_hidden_size+self.node_auxiliary_size)] * (q_max_length - len(v))).to(device))) for v in q_js_encoding])
        # now use GCN:
        degree_mat = torch.Tensor(degree_mat).to(device)
        A_hat = batch_normalize(degree_mat)

        output_1 = self.gcn1(q_js_encoding)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(A_hat, output_1))  # [B, N, N], [B, N, Hid_C]

        output_2 = self.gcn2(output_1)
        q_real_out = self.act(torch.matmul(A_hat, output_2))

        q_input = torch.stack([self.encoding_input(v) for v in q_input_seq]).to(device)
        # print(q_real_out.size())
        q_real_out = q_real_out.unsqueeze(1)
        # print(q_real_out.size())
        # q_real_out = q_real_out.view((batch_size, -1))
        q_real_out = self.graph_learner(q_real_out)
        q_real_out = q_real_out.view((batch_size, -1))
        # print(q_real_out.size())
        # print(q_real_out.size())

        # for index
        index_encoding = []
        for i in range(batch_size):
            if len(index_info[i]) > 0:
                current_encode = self.encoding_index(index_info[i])
                if len(current_encode) > self.max_index_count:
                    current_encode = current_encode[:self.max_index_count]
                index_encoding.append(current_encode)
            else:
                index_encoding.append(torch.FloatTensor([[0] *
                                                         (self.keyword_embedding_size * 2 +
                                                          self.index_info_size - 2)]).to(device))
        index_encoding = torch.stack(
            [torch.cat((v, torch.FloatTensor(
                [[0] * (self.keyword_embedding_size * 2 + self.index_info_size - 2)] *
                (self.max_index_count - len(v))).to(device))) for v in index_encoding])
        temp = index_encoding.unsqueeze(1).to(torch.float32)
        temp2 = index_encoding.unsqueeze(1).to(torch.float32)
        # print(index_encoding.size())
        # print(temp2.size())
        index_wise_attention = self.index_wise_attention_cnn(temp)
        # print(index_wise_attention.size())
        index_wise_attention = index_wise_attention.view((batch_size, -1))
        # print(index_wise_attention.size())
        index_wise_attention = self.index_wise_attention_Linear(index_wise_attention)
        index_wise_attention = index_wise_attention.view((batch_size, self.max_index_count, 1))
        element_wise_attention = self.element_wise_attention_cnn(temp2)
        # print(element_wise_attention.size())
        element_wise_attention = element_wise_attention.view((batch_size, -1))
        # print(element_wise_attention.size())
        element_wise_attention = self.element_wise_attention_Linear(element_wise_attention)
        element_wise_attention = element_wise_attention.view(
            (batch_size, 1, self.keyword_embedding_size * 2 + self.index_info_size - 2))
        index_out = index_encoding.mul(index_wise_attention).mul(element_wise_attention)
        index_out = index_out.to(torch.float32)
        index_out = self.index_learner(index_out)
        index_out = index_out.view((batch_size,  -1))


        features_tensor = torch.cat([q_input, q_real_out, index_out], dim=1).to(device)
        features_tensor = torch.add(features_tensor,self.resnet1(features_tensor)).to(device)

        out = self.regressor(features_tensor).squeeze()
        return out


def train(data,keywords,usekeyword=True,usestring=True,usesequence=True,model_pre=None,result_file=None):
    train_rate = 0.9
    batch_size = 512
    epoch = 50
    lr = 0.001
    # lam = 0.01
    weight_decay = 1e-5
    args = {}
    args['keywords'] = keywords
    args['keyword_embedding_size'] = 32
    args['char_embedding_size'] = 64
    args['node_auxiliary_size'] = 3
    args['first_hidden_size'] = 64
    args['second_hidden_size'] = 128
    args['drop_rate'] = 0.2
    args['other_size'] = len(data[0][0][-1])
    args['key2index'] = {word:index for index,word in enumerate(keywords)}
    args['index2key'] = {value:key for key,value in args['key2index'].items()}
    args['usekeyword'] = usekeyword
    args['usestring'] = usestring
    args['usesequence'] = usesequence
    args['index_info_size'] = 17
    args['q_max_len'] = 50
    args['graph_learner_size'] = 77
    args['gcn_out_size'] = 32

    max_index_count = 0
    for item in data:
        if len(item[0][3]) > max_index_count:
            max_index_count = len(item[0][3])
    print('max_index_count', max_index_count)
    args['max_index_count'] = max_index_count

    if model_pre is not None:
        model_pre += "{}_{}_{}_{}_{}-{}-{}-{}-{}-{}-{}.model".format(usekeyword,usestring,usesequence,lr,weight_decay,args['keyword_embedding_size'],args['char_embedding_size'],
                                                                   args['node_auxiliary_size'],args['first_hidden_size'],args['second_hidden_size'],args['drop_rate'])
    for i in range(len(data)):
        data[i] = list(data[i])
        data[i][1] = np.log(data[i][1])
    #
    t = np.array([i[1] for i in data[:-2000]])
    d_mean = np.mean(t)
    d_std = np.std(t)
    d_max = np.max(t)
    d_min = np.min(t)
    label_statistic = [d_mean, d_std]
    print('mean: {}, std: {}, max: {}, min: {}'.format(d_mean, d_std, d_max, d_min))
    np.save('label_statistic.npy', label_statistic)
    # label_statistic = np.load('./label_statistic.npy')
    # d_mean = label_statistic[0]
    # d_std = label_statistic[1]
    for k in range(len(data)):
        data[k][1] = (data[k][1] - d_mean) / d_std
    # print((np.max(t) - d_mean) / d_std)
    # exit()


    train_data = data[:-2000]
    validate_data = data[-2000:-1000]
    test_data = data[-1000:]
    validate_X = [data[0] for data in validate_data]
    validate_Y = torch.FloatTensor(np.array([data[1] for data in validate_data]))
    # validate_Y = torch.FloatTensor(np.log([data[1] for data in validate_data]))
    validate_Y = validate_Y.to(device)
    gcncost = GCNCost(args)
    gcncost = gcncost.to(device)
    mape = mape_loss()
    mae = mae_loss()
    choice = 0
    learning_rate = lr
    pre_validate_loss = 0.06826
    # pre_train_loss = 100




    step_per_epoch = int(np.ceil(len(train_data)/batch_size))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(gcncost.parameters(), lr=lr, weight_decay=weight_decay)
    if model_pre is not None and os.path.exists(model_pre):
        check_point = torch.load(model_pre)
        gcncost.load_state_dict(check_point['model'])
        optimizer.load_state_dict(check_point['optimizer'])
        print('prelimary model loaded.')
        # pre_validate_loss = None

    if choice == 0:
        pass
    else:
        for i in range(epoch):
            # decay = (i + 1) // 2
            if learning_rate > 0.0000001:
                learning_rate = lr * pow(0.8, i)#0.8
            if learning_rate < 0.0000001:
                learning_rate = 0.0000001
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            random.shuffle(train_data)
            gcncost.train()
            for step in range(step_per_epoch):
                train_per_step = train_data[step*batch_size:(step+1)*batch_size]

                X = [data[0] for data in train_per_step]
                # Y = torch.FloatTensor(np.log([data[1] for data in train_per_step]))
                Y = torch.FloatTensor(np.array([data[1] for data in train_per_step]))
                Y = Y.to(device)
                Y_ = gcncost(X)
                loss = loss_func(Y_,Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Epoch: {}, step: {}, learning rate: {}, train loss: {}'.format(i, step, learning_rate,  loss,))


            gcncost.eval()
            validate_Y_ = gcncost(validate_X)
            validate_loss = loss_func(validate_Y_,validate_Y)
            print('Epoch: {}, step: {}, learning rate: {} test lost: {}, train loss: {}'\
                  .format(i,step,learning_rate,validate_loss,loss,))
            if pre_validate_loss > validate_loss: #and pre_validate_loss > validate_loss.item():
                if model_pre is not None:
                    check_point = {'model': gcncost.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(check_point, model_pre)
                    print('new best saved')
                    pre_validate_loss = validate_loss.item()


    if result_file is not None:
        results = []
        gcncost.eval()
        with open(result_file,'w',encoding='utf-8') as w:
            myerror = []
            for d in test_data:
                x,y,l = d
                with torch.no_grad():
                    y_ = gcncost([x])#.squeeze()
                    y_ = y_ * d_std + d_mean
                y = y * d_std + d_mean
                y = np.exp(y)
                mae_ = mae(torch.exp(y_),torch.FloatTensor([y]).to(device)).cpu()
                mape_ = mape(torch.exp(y_),torch.FloatTensor([y]).to(device)).cpu()
                y_ = y_.cpu()
                __y = np.exp(y_.squeeze().detach().numpy())

                if(__y > y):
                    myerror.append(__y/(y*1.0) )
                else:
                    myerror.append((y)/(__y*1.0))
                results.append((l, np.exp(y_.squeeze().detach().numpy()), y, mae_.squeeze().detach().numpy().flatten()[0], mape_.squeeze().detach().numpy().flatten()[0]))
            myerror.sort()
            myerror = np.array(myerror)
            # print(myerror)
            print('median: ', np.median(myerror),' max: ', np.max(myerror), ' mean: ', np.mean(myerror))
            results = sorted(results,key=lambda x:x[3],reverse=True)
            for r in results:
                w.write("{}\n".format(r))

from DataProcess import process_data
if __name__ == "__main__":
    if not os.path.exists('./ProgramTemp/data.pickle'):
        process_data()
        data,keywords = build_test()
        random.shuffle(data)
        pickle.dump(data, open('./ProgramTemp/data.pickle', 'wb'))
        pickle.dump(keywords, open('./ProgramTemp/keywords.pickle', 'wb'))
    else:
        data = pickle.load(open('./ProgramTemp/data.pickle', 'rb'))
        keywords = pickle.load(open('./ProgramTemp/keywords.pickle', 'rb'))
    temp1 = []
    for i in data[:-2000]:
        for j in i[0][1]:
            temp1.append(j)
    temp1 = np.array(temp1)
    mean1 = np.mean(temp1, axis=0)
    std1 = np.std(temp1, axis=0)
    max1 = np.max(temp1, axis=0)
    # min1 = np.min(temp1, axis=0)
    data_statistic = [mean1, std1]
    # data_statistic2 = np.load('./data_statistic.npy')
    # mean1 = data_statistic2[0]
    # std1 = data_statistic2[1]
    # print(data_statistic-data_statistic2)
    np.save('data_statistic.npy', data_statistic)
    for i in range(len(data)):
        for j in range(len(data[i][0][1])):
            data[i][0][1][j][0] = (data[i][0][1][j][0] - mean1[0]) / std1[0]  # (max1[0] - min1[0])
            data[i][0][1][j][1] = (data[i][0][1][j][1] - mean1[1]) / std1[1]  # (max1[1] - min1[1])
            data[i][0][1][j][2] = (data[i][0][1][j][2] - mean1[2]) / std1[2]  # (max1[2] - min1[2])
    print('done. data length: ', len(data))
    pro_dir = "./ProgramTemp/"
    wide_deep_file_pre = "GCN_Attention_Estimator_"
    result = "gcn_attention_result.text"
    train(data, keywords,model_pre=pro_dir+wide_deep_file_pre,result_file=pro_dir+result)