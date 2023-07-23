import time
import MINST,CIFAR10
import torch
import torch.nn as nn
from BatchIndex import BatchIndex
from torch.nn import Sequential
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def dicadd(dic1,dic2):
    dic={}
    if not dic1:
        return dic2
    if not dic2:
        return dic1
    for key in dic1.keys():
        dic[key]=dic1[key]+dic2[key]
    return dic

def dicmul(dic1,x):
    for key in dic1.keys():
        dic1[key]=dic1[key]*x
    return dic1

def myLoadt(begin_index,end_index,lst):
    lst1=[lst[i][0] for i in range(begin_index,end_index)]
    re=torch.stack(lst1)
    return re

def myLoadi(begin_index,end_index,lst):
    lst1=[lst[i][1] for i in range(begin_index,end_index)]
    re=torch.tensor(lst1)
    return re

def data_to_device(name,device,show_time=False):
    t=time.time()
    if name=="MINST":
        train_data, test_data = MINST.getData()
    elif name=="CIFAR10":
        train_data, test_data = CIFAR10.getData()
    else:
        raise Exception("Only have data 'MINST', 'CIFAR10'")
    if show_time:
        print('begin time', time.time() - t)
    muti = 1
    train_size = len(train_data)
    test_size = len(test_data)
    train_index = train_size
    test_index = test_size
    start_train = [int(i * train_index / muti) for i in range(muti)]
    end_train = [int(i * train_index / muti) for i in range(1, muti + 1)]
    start_test = [int(i * test_index / muti) for i in range(muti)]
    end_test = [int(i * test_index / muti) for i in range(1, muti + 1)]
    with Pool() as pool:
        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        for i in range(muti):
            re1 = pool.apply_async(myLoadt, (start_train[i], end_train[i], train_data))
            re2 = pool.apply_async(myLoadi, (start_train[i], end_train[i], train_data))
            re3 = pool.apply_async(myLoadt, (start_test[i], end_test[i], test_data))
            re4 = pool.apply_async(myLoadi, (start_test[i], end_test[i], test_data))
            lst1.append(re1)
            lst2.append(re2)
            lst3.append(re3)
            lst4.append(re4)
        if show_time:
            print('finish appoint', time.time() - t)
        pool.close()
        pool.join()
        lst1 = [item.get() for item in lst1]
        lst2 = [item.get() for item in lst2]
        lst3 = [item.get() for item in lst3]
        lst4 = [item.get() for item in lst4]
    if show_time:
        print('pool time', time.time() - t)

    X_train = torch.concat(lst1)
    X_test = torch.concat(lst3)
    y_train = torch.concat(lst2)
    y_test = torch.concat(lst4)
    X_traint = X_train.to(device)
    y_traint = y_train.to(device)
    X_testt = X_test.to(device)
    y_testt = y_test.to(device)
    print(X_traint.shape, y_testt.shape)
    if show_time:
        print('load data:', time.time() - t)
    return [X_traint,y_traint,X_testt,y_testt],train_size,test_size

def drawpic(path_of_inf,lossn='1.png',accn='2.png',dpi=480,show=True):
    df = pd.read_csv(path_of_inf)
    lenth=df.shape[0]
    x=np.arange(1,lenth+1)
    plt.plot(x, df.iloc[:,1], label='training Loss')
    plt.plot(x,  df.iloc[:,3], label='test Loss')
    plt.legend()
    plt.savefig(lossn, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.cla()
    plt.plot(x,  df.iloc[:,2], label='training acc')
    plt.plot(x,  df.iloc[:,4], label='test acc')
    plt.legend()
    plt.savefig(accn, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.cla()

class CNN(nn.Module):
    def __init__(self,width,lenth,channel):
        super(CNN,self).__init__()
        self.width=width
        self.lenth=lenth
        self.channel=channel
        self.conv=Sequential(
            nn.Conv2d(channel,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dense=Sequential(
            nn.Linear(8*int(self.width/2)*int(self.lenth/2),128),
            nn.ReLU(),
            nn.Linear(128,10),
        )
        print('CNN model created')

    def forward(self,x):
        x=self.conv(x)
        x=x.view(-1,8*int(self.width/2)*int(self.lenth/2))
        x=self.dense(x)
        return x

class Trainer:
    def __init__(self,load_data='MINST',data_L=None,batch_size=64,bacth_size1=512,optimizer=torch.optim.Adam,para_dict=None,
                 loss_F=nn.CrossEntropyLoss()):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if load_data:
            data_L,self.train_size,self.test_size=data_to_device(load_data,self.device)
            self.X_traint, self.y_traint, self.X_testt, self.y_testt = data_L
        else:
            if not data_L:
                raise Exception("load_data and data_L can not be empty meanwhile!")
            self.X_traint, self.y_traint, self.X_testt, self.y_testt = data_L
            self.train_size=self.y_traint.shape[0]
            self.test_size=self.y_testt.shape[0]

        channel,lenth,width=self.X_traint.shape[1:]
        self.model = CNN(width, lenth, channel).to(self.device)
        if not para_dict:
            para_dict={}
        self.optimizer=optimizer(self.model.parameters(),**para_dict)
        self.batch_size=batch_size
        self.batch_size1=bacth_size1
        self.loss_F=loss_F
        self.train_lossL=[]
        self.train_accuracyL=[]
        self.test_lossL=[]
        self.test_accuracyL=[]
        self.recordL=[self.train_lossL,self.train_accuracyL,self.test_lossL,self.test_accuracyL]

    def clc_test(self):
        self.test_lossL=[]
        self.test_accuracyL=[]

    def draw(self,train,test,show=True,save_loss='loss.png',save_accuracy='accuracy.png',dpi=480):
        if train and test:
            lenth1=len(self.train_lossL)
            lenth2=len(self.test_lossL)
            x1=np.arange(lenth1)
            x2=np.arange(lenth2)
            plt.plot(x1,self.train_lossL,label='training loss')
            plt.plot(x2,self.test_lossL,label='testing loss')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("loss")
            plt.title("loss changing in training rounds")
            if save_loss:
                plt.savefig(save_loss,dpi=dpi)
            if show:
                plt.show()
            plt.cla()
            plt.plot(x1,self.train_accuracyL,label='training accuracy')
            plt.plot(x2,self.test_accuracyL,label='testing accuracy')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("accuracy(%)")
            plt.title("accuracy changing in training rounds")
            if save_accuracy:
                plt.savefig(save_accuracy,dpi=dpi)
            if show:
                plt.show()
            plt.cla()
        elif train:
            lenth1 = len(self.train_lossL)
            x1 = np.arange(lenth1)
            plt.plot(x1, self.train_lossL, label='training loss')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("loss")
            plt.title("loss changing in training rounds")
            if save_loss:
                plt.savefig(save_loss,dpi=dpi)
            if show:
                plt.show()
            plt.cla()
            plt.plot(x1, self.train_accuracyL, label='training accuracy')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("accuracy(%)")
            plt.title("accuracy changing in training rounds")
            if save_accuracy:
                plt.savefig(save_accuracy,dpi=dpi)
            if show:
                plt.show()
            plt.cla()
        elif test:
            lenth2 = len(self.test_lossL)
            x2 = np.arange(lenth2)
            plt.plot(x2, self.test_lossL, label='testing loss')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("loss")
            plt.title("loss changing in training rounds")
            if save_loss:
                plt.savefig(save_loss,dpi=dpi)
            if show:
                plt.show()
            plt.cla()
            plt.plot(x2, self.test_accuracyL, label='testing accuracy')
            plt.legend()
            plt.xlabel("communication rounds")
            plt.ylabel("accuracy(%)")
            plt.title("accuracy changing in training rounds")
            if save_accuracy:
                plt.savefig(save_accuracy,dpi=dpi)
            if show:
                plt.show()
            plt.cla()



    def test(self,test_dataL=None):
        if test_dataL:
            X_testt,y_testt=test_dataL
            test_size=y_testt.shape[0]
        else:
            X_testt, y_testt = self.X_testt,self.y_testt
            test_size=self.test_size
        self.model.eval()
        testing_loss = 0.0
        testing_correct = 0.0
        for a, b in BatchIndex(test_size, self.batch_size, False):
            X_test, y_test = X_testt[a:b], y_testt[a:b]
            outputs = self.model(X_test)
            loss = self.loss_F(outputs, y_test)
            testing_loss += loss.item() * (b - a)
            _, pred = torch.max(outputs, 1)
            testing_correct += torch.sum(torch.eq(pred, y_test))
            # print(testing_correct)

        teL = testing_loss / test_size
        teA = 100 * testing_correct / test_size
        print("Test Loss: {:.4f}  Test Accuracy: {:.4f}%".format(teL,teA))
        self.test_lossL.append(teL)
        self.test_accuracyL.append(teA.item())
        return teL,teA.item()


    def evolve(self,show_d=True):
        running_loss = 0.0 
        running_correct = 0.0  
        self.model.train()
        for a,b in BatchIndex(self.train_size,self.batch_size,True):
            X_train, y_train = self.X_traint[a:b],self.y_traint[a:b]
            outputs = self.model(X_train)
            _, pred = torch.max(outputs.data, 1)
            self.optimizer.zero_grad()
            loss = self.loss_F(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*(b-a)
            running_correct += torch.sum(torch.eq(pred, y_train))

        self.model.eval()
        testing_loss = 0.0
        testing_correct = 0.0
        for a,b in BatchIndex(self.test_size,self.batch_size,False):
            X_test, y_test = self.X_testt[a:b], self.y_testt[a:b]
            outputs = self.model(X_test)
            loss = self.loss_F(outputs, y_test)
            testing_loss += loss.item() * (b - a)
            _, pred = torch.max(outputs, 1)
            testing_correct += torch.sum(torch.eq(pred, y_test))
            # print(testing_correct)
        trL=running_loss / self.train_size
        trA=100 * running_correct / self.train_size
        teL=testing_loss / self.test_size
        teA=100 * testing_correct / self.test_size
        if show_d:
            print("Loss: {:.4f}  Train Accuracy: {:.4f}%   Test Loss: {:.4f} Test Accuracy: {:.4f}%".format(trL,trA,teL,teA))
        self.train_lossL.append(trL)
        self.train_accuracyL.append(trA.item())
        self.test_lossL.append(teL)
        self.test_accuracyL.append(teA.item())

    def save(self,path_of_weights,path_of_inf):
        torch.save(self.model, path_of_weights)
        col_name=['test loss','test accuracy','train loss','train accuracy']
        df=pd.DataFrame()
        df['epcho']=list(range(1,len(self.train_lossL)+1))
        for i in range(4):
            df[col_name[i]]=self.recordL[i]

        df.to_csv(path_of_inf,index=False)

    def load(self,path_of_weights,path_of_inf):
        checkpoint = torch.load(path_of_weights)
        self.model.load_state_dict(checkpoint.state_dict())
        df=pd.read_csv(path_of_inf)
        for i in range(1,5):
            self.recordL[i-1][:]=list(df.iloc[:,i])[:]


    def save_weights(self,path):
        torch.save(self.model,path)

    def load_weights(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint.state_dict())

class Client(Trainer):
    def __init__(self,train_dataL,batch_size=64,bacth_size1=512,optimizer=torch.optim.Adam,para_dict=None,
                 loss_F=nn.CrossEntropyLoss(),client_epchos=5):
        train_dataL=list(train_dataL)
        train_dataL.append(train_dataL[0])
        train_dataL.append(train_dataL[1])
        super(Client, self).__init__('',train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F)
        self.acclast=0
        self.client_epchos=client_epchos

    def client_round(self,newpara=None,show_detial=True):
        if newpara:
            self.model.load_state_dict(newpara)
        params1 = {}
        for key in self.model.state_dict().keys():
            params1[key]=self.model.state_dict()[key].clone()
        for i in range(self.client_epchos):
            self.evolve(show_d=show_detial)
        params2=self.model.state_dict()
        params_diff = {}
        for key in params1.keys():
            params_diff[key] = params2[key] - params1[key]
            #print(params_diff[key])
        return params_diff

    def evolve(self,show_d=True):
        running_loss = 0.0
        running_correct = 0.0 
        self.model.train()
        for a,b in BatchIndex(self.train_size,self.batch_size,True):
            X_train, y_train = self.X_traint[a:b],self.y_traint[a:b]
            outputs = self.model(X_train)
            _, pred = torch.max(outputs.data, 1)
            self.optimizer.zero_grad()
            loss = self.loss_F(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*(b-a) 
            running_correct += torch.sum(torch.eq(pred, y_train))

        trL=running_loss / self.train_size
        trA=100 * running_correct / self.train_size
        if show_d:
            print("Loss: {:.4f}  Train Accuracy: {:.4f}%".format(trL,trA))
        self.train_lossL.append(trL)
        self.train_accuracyL.append(trA.item())

class Server:
    def __init__(self,pretrained_data):
        self.param=pretrained_data

    def server_round(self,client_updateL):
        agg_param=client_updateL[0]

        lenth=len(client_updateL)
        if lenth>1:
            for key in agg_param.keys():
                for para_dict in client_updateL[1:]:
                    agg_param[key] += para_dict[key]
                agg_param[key] = agg_param[key] / lenth
                self.param[key]=agg_param[key]+self.param[key]
        else:
            for key in agg_param.keys():
                self.param[key] = agg_param[key] + self.param[key]
                #print(agg_param[key])

if __name__ == '__main__':
    t=time.time()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_data, test_data = MINST.getData()
    muti = 2
    train_size = len(train_data)
    test_size = len(test_data)
    train_index = train_size
    test_index = test_size
    start_train = [int(i * train_index / muti) for i in range(muti)]
    end_train = [int(i * train_index / muti) for i in range(1, muti + 1)]
    start_test = [int(i * test_index / muti) for i in range(muti)]
    end_test = [int(i * test_index / muti) for i in range(1, muti + 1)]
    print(end_test)
    with Pool() as pool:
        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        for i in range(muti):
            re1 = pool.apply_async(myLoadt, (start_train[i], end_train[i], train_data))
            re2 = pool.apply_async(myLoadi, (start_train[i], end_train[i], train_data))
            re3 = pool.apply_async(myLoadt, (start_test[i], end_test[i], test_data))
            re4 = pool.apply_async(myLoadi, (start_test[i], end_test[i], test_data))
            lst1.append(re1)
            lst2.append(re2)
            lst3.append(re3)
            lst4.append(re4)
        print('finish appoint', time.time() - t)
        pool.close()
        pool.join()
        lst1 = [item.get() for item in lst1]
        lst2 = [item.get() for item in lst2]
        lst3 = [item.get() for item in lst3]
        lst4 = [item.get() for item in lst4]
    print('pool time', time.time() - t)

    X_train = torch.concat(lst1)
    X_test = torch.concat(lst3)
    y_train = torch.concat(lst2)
    y_test = torch.concat(lst4)
    X_traint = X_train.to(device)
    y_traint = y_train.to(device)
    X_testt = X_test.to(device)
    y_testt = y_test.to(device)
    print(X_traint.shape, y_testt.shape)
    print('load data:', time.time() - t)
    totalL=[X_traint,y_traint,X_testt,y_testt]


    client_num=10
    communication_rounds=15
    client_epchos=1

    ind=torch.randperm(train_size)
    per_count=(train_size/client_num)
    clientL=[]
    for i in range(client_num):
        dataL=[]
        dataL.append(X_traint[ind[int(per_count*i):int(per_count*(i+1))]])
        dataL.append(y_traint[ind[int(per_count*i):int(per_count*(i+1))]])
        client=Client(dataL,client_epchos=client_epchos)
        clientL.append(client)

    test_trainer=Trainer(load_data='',data_L=totalL)
    initial_dict=test_trainer.model.state_dict()
    server=Server(initial_dict)
    for i in range(communication_rounds):
        print("round {}/{}".format(i,communication_rounds))
        client_updateL=[]
        for client in clientL:
            client_update_dic=client.client_round(server.param,show_detial=False)
            client_updateL.append(client_update_dic)
        #print(client_updateL[0][list(server.param.keys())[0]])
        server.server_round(client_updateL)
        test_trainer.model.load_state_dict(server.param)
        #print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
    test_trainer.draw(0,1,save_loss='normal_FL_loss.png',save_accuracy='normal_FL_accruacy.png')
