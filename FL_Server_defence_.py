# 在FL_Server_defence.py的基础上给Siren添加了flag机制

import random
from FL_Base import *
import FL_Client as mc

class Krum(Server):
    def __init__(self,pretrained_data,client_num,malicious_client_num):
        super(Krum,self).__init__(pretrained_data)
        self.client_num=client_num
        self.malicious_client_num=malicious_client_num

    def server_round(self,client_updateL):
        flattened_tensorL=[]
        distance_matrix=torch.zeros((self.client_num,self.client_num))
        for i in range(len(client_updateL)):
            tens = list(client_updateL[i].values())
            flattened_tensor = torch.cat([t.flatten() for t in tens])
            for j in range(len(flattened_tensorL)):
                dis=torch.sum((flattened_tensorL[j]-flattened_tensor)**2)
                distance_matrix[i,j]=distance_matrix[j,i]=dis
            flattened_tensorL.append(flattened_tensor)
        distance_matrix,_=torch.sort(distance_matrix,dim=1)
        distance_matrix=torch.sum(distance_matrix[:,:self.client_num-self.malicious_client_num-2],dim=1)
        ind=torch.argmin(distance_matrix)
        client_updateL=[client_updateL[ind]]

        super().server_round(client_updateL)

class Coordinate_wise_Median(Server):
    def __init__(self,pretrained_data):
        super(Coordinate_wise_Median,self).__init__(pretrained_data)

    def server_round(self, client_updateL):
        #print(len(client_updateL))
        for key in self.param.keys():
            dt=torch.stack([client_update[key] for client_update in client_updateL])
            #print(dt.shape)
            dt,_=torch.median(dt,dim=0)
            self.param[key]=self.param[key]+dt

class FLTrust(Server):
    def __init__(self,pretrained_data,root_train_data,client_epchos=5):
        super(FLTrust,self).__init__(pretrained_data)
        self.cli=Client(root_train_data,client_epchos=client_epchos)

    def server_round(self,client_updateL):
        base_update=self.cli.client_round(self.param)
        fl_base=torch.cat([ten.flatten() for ten in base_update.values()])
        tsL=[]
        normalize_factorL=[]
        for client_update in client_updateL:
            fl_tensor=torch.cat([ten.flatten() for ten in client_update.values()])
            L21=torch.sum(fl_tensor**2)**0.5
            L22=torch.sum(fl_base**2)**0.5
            cos_sim=torch.sum(fl_tensor*fl_base)/(L22*L21)
            normalize_factor=L22/L21
            print(cos_sim,normalize_factor)
            cos_sim=cos_sim.item()
            tsL.append((cos_sim+abs(cos_sim))/2)
            normalize_factorL.append(normalize_factor)
        s=sum(tsL)
        for key in base_update.keys():
            tmp=tsL[0]*client_updateL[0][key]*normalize_factorL[0]
            for i in range(1,len(tsL)):
                tmp=tsL[i]*client_updateL[i][key]*normalize_factorL[i]+tmp
            self.param[key]=self.param[key]+tmp/s

class Siren(Server):
    def __init__(self,pretrained_data,root_train_data,acc_control=10,db=False):
        super(Siren,self).__init__(pretrained_data)
        self.paraml=dict(self.param)
        #self.cli=Client(root_train_data,client_epchos=client_epchos)
        self.acc_control=acc_control
        self.test_trainer=Trainer('',[root_train_data[0],root_train_data[1],root_train_data[0],root_train_data[1]])
        self.db=db
        self.alarm_flag=False

    def server_round(self,client_updateL,alarmL):
        if True in alarmL:
            print('alarm List',alarmL)
            acc_alarm=torch.empty((2,sum(alarmL)))
            ind_alarm=0
            for i in range(len(alarmL)):
                if alarmL[i]:
                    paramt=dicadd(self.param,client_updateL[i])
                    self.test_trainer.model.load_state_dict(paramt)
                    _,acc=self.test_trainer.test()
                    acc_alarm[0,ind_alarm]=acc
                    acc_alarm[1,ind_alarm]=i
                    ind_alarm+=1
            accmax=torch.max(acc_alarm[0])
            s=torch.sum((accmax-self.acc_control)>acc_alarm[0])
            print('s',s)

            if s:#情况4
                print('情况4')
                count = 0
                paramdt={}
                usedL=[]
                for i in range(sum(alarmL)):
                    if acc_alarm[0,i]>accmax-self.acc_control:
                        count+=1
                        paramdt=dicadd(paramdt,client_updateL[int(acc_alarm[1,i])])
                        usedL.append(acc_alarm[1,i].item())
                if count:
                    dicmul(paramdt, 1 / count)
                    if not self.alarm_flag:
                        print('---------using last----------')
                        self.param = dicadd(self.paraml, paramdt)
                        self.alarm_flag=True
                    else:
                        print('---------using now----------')
                        self.paraml=self.param
                        self.param = dicadd(self.param, paramdt)
                        self.alarm_flag = False
                # debug
                if self.db:
                    print('------debug-------')
                    for i in range(len(alarmL)):
                        if alarmL[i]:
                            paramt = dicadd(self.paraml, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
            else:
                silent_num=len(alarmL)-sum(alarmL)
                if silent_num:
                    acc_silent = torch.empty((2, silent_num))
                    ind_alarm = 0
                    for i in range(len(alarmL)):
                        if not alarmL[i]:
                            paramt = dicadd(self.param, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
                            acc_silent[0, ind_alarm] = acc
                            acc_silent[1, ind_alarm] = i
                            ind_alarm += 1
                    accsmax = torch.max(acc_silent[0])
                if silent_num and accsmax+self.acc_control>accmax:#情况3+假警报,认为真诚客户端均silent
                    print('情况3+假警报,认为真诚客户端均silent')
                    count=0
                    paramdt={}
                    usedL=[]
                    for i in range(len(alarmL)-sum(alarmL)):
                        if acc_silent[0,i]>accsmax-self.acc_control:
                            paramdt = dicadd(paramdt, client_updateL[int(acc_silent[1, i])])
                            count+=1
                            usedL.append(acc_silent[1,i].item())
                    if count:
                        print('---------using now----------')
                        dicmul(paramdt,1/count)
                        self.paraml=self.param
                        self.param=dicadd(self.param,paramdt)
                        self.alarm_flag=False
                else:#情况3+真警报,认为所有警报客户端为真诚客户端
                    print('情况3+真警报,认为所有警报客户端为真诚客户端')
                    count=sum(alarmL)
                    paramdt={}
                    usedL=[]
                    for i in range(count):
                        paramdt=dicadd(paramdt,client_updateL[int(acc_alarm[1,i])])
                        usedL.append(acc_alarm[1,i].item())
                    if count:
                        dicmul(paramdt, 1 / count)
                        if not self.alarm_flag:
                            print('---------using last----------')
                            self.param = dicadd(self.paraml, paramdt)
                            self.alarm_flag = True
                        else:
                            print('---------using now----------')
                            self.paraml = self.param
                            self.param = dicadd(self.param, paramdt)
                            self.alarm_flag = False
                    if self.db:
                        print('------debug-------')
                        for i in range(len(alarmL)):
                            if alarmL[i]:
                                paramt = dicadd(self.paraml, client_updateL[i])
                                self.test_trainer.model.load_state_dict(paramt)
                                _, acc = self.test_trainer.test()
            print(usedL,'be added')
        else:#情况1和2
            print('情况1和2')
            self.paraml=self.param
            super().server_round(client_updateL)
            self.alarm_flag=False

class Siren_fltrust(Siren):
    def __init__(self, pretrained_data, root_train_data, acc_control=10,db=False):
        super(Siren_fltrust,self).__init__(pretrained_data, root_train_data, acc_control,db)




    def server_round(self,client_updateL,alarmL):
        if True in alarmL:
            print('alarm List',alarmL)
            acc_alarm=torch.empty((2,sum(alarmL)))
            ind_alarm=0
            for i in range(len(alarmL)):
                if alarmL[i]:
                    paramt=dicadd(self.param,client_updateL[i])
                    self.test_trainer.model.load_state_dict(paramt)
                    _,acc=self.test_trainer.test()
                    acc_alarm[0,ind_alarm]=acc
                    acc_alarm[1,ind_alarm]=i
                    ind_alarm+=1

            maxind=torch.argmax(acc_alarm[0])
            accmax=acc_alarm[0,maxind]
            maxtensor_alarm=torch.cat([ten.flatten() for ten in client_updateL[int(acc_alarm[1,maxind])].values()])
            s=torch.sum((accmax-self.acc_control)>acc_alarm[0])
            print('s',s)

            if s:#情况4
                print('情况4')
                count = 0
                paramdt={}
                usedL=[]
                for i in range(sum(alarmL)):
                    if acc_alarm[0,i]>accmax-self.acc_control and cos_analysis(maxtensor_alarm,client_updateL[int(acc_alarm[1,i])]):
                        count+=1
                        paramdt=dicadd(paramdt,client_updateL[int(acc_alarm[1,i])])
                        usedL.append(acc_alarm[1,i].item())
                if count:
                    dicmul(paramdt, 1 / count)
                    if not self.alarm_flag:
                        print('---------using last----------')
                        self.param = dicadd(self.paraml, paramdt)
                        self.alarm_flag = True
                    else:
                        print('---------using now----------')
                        self.paraml = self.param
                        self.param = dicadd(self.param, paramdt)
                        self.alarm_flag = False
                #debug
                if self.db:
                    print('------debug-------')
                    for i in range(len(alarmL)):
                        if alarmL[i]:
                            paramt = dicadd(self.paraml, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
            else:
                silent_num=len(alarmL)-sum(alarmL)
                if silent_num:
                    acc_silent = torch.empty((2, silent_num))
                    ind_alarm = 0
                    for i in range(len(alarmL)):
                        if not alarmL[i]:
                            paramt = dicadd(self.param, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
                            acc_silent[0, ind_alarm] = acc
                            acc_silent[1, ind_alarm] = i
                            ind_alarm += 1
                    maxsind=torch.argmax(acc_silent[0])
                    accsmax = acc_silent[0,maxsind]
                    maxtensor_slient = torch.cat(
                        [ten.flatten() for ten in client_updateL[int(acc_silent[1, maxsind])].values()])
                if silent_num and accsmax+self.acc_control>accmax:#情况3+假警报,认为真诚客户端均silent
                    print('情况3+假警报,认为真诚客户端均silent')
                    count=0
                    paramdt={}
                    usedL=[]
                    for i in range(len(alarmL)-sum(alarmL)):
                        if acc_silent[0,i]>accsmax-self.acc_control and cos_analysis(maxtensor_slient,client_updateL[int(acc_silent[1,i])]):
                            paramdt = dicadd(paramdt, client_updateL[int(acc_silent[1, i])])
                            count+=1
                            usedL.append(acc_silent[1,i].item())
                    if count:
                        print('---------using now----------')
                        dicmul(paramdt,1/count)
                        self.paraml=self.param
                        self.param=dicadd(self.param,paramdt)
                        self.alarm_flag=False
                else:#情况3+真警报,认为所有警报客户端为真诚客户端
                    print('情况3+真警报,认为所有警报客户端为真诚客户端')
                    count=sum(alarmL)
                    paramdt={}
                    usedL=[]
                    for i in range(count):
                        paramdt=dicadd(paramdt,client_updateL[int(acc_alarm[1,i])])
                        usedL.append(acc_alarm[1,i].item())
                    if count:
                        dicmul(paramdt, 1 / count)
                        if not self.alarm_flag:
                            print('---------using last----------')
                            self.param = dicadd(self.paraml, paramdt)
                            self.alarm_flag = True
                        else:
                            print('---------using now----------')
                            self.paraml = self.param
                            self.param = dicadd(self.param, paramdt)
                            self.alarm_flag = False
                    if self.db:
                        print('------debug-------')
                        for i in range(len(alarmL)):
                            if alarmL[i]:
                                paramt = dicadd(self.paraml, client_updateL[i])
                                self.test_trainer.model.load_state_dict(paramt)
                                _, acc = self.test_trainer.test()
            print(usedL,'be added')
        else:#情况1和2
            print('情况1和2')
            self.paraml=self.param
            super(Siren,self).server_round(client_updateL)
            self.alarm_flag=False


class Siren_PA(Siren):
    def __init__(self, pretrained_data, root_train_data, client_num,acc_control=10,Cp=4.5,Ca=0.5,db=False):
        '''
        :param pretrained_data:
        :param root_train_data:
        :param client_num:
        :param acc_control:
        :param Cp: 惩罚阈值(超出该值进行惩罚)
        :param Ca: 每回合表现好的惩罚阈值减少量
        '''
        super(Siren_PA,self).__init__(pretrained_data, root_train_data, acc_control,db)
        self.malious_indexL=torch.zeros(client_num)
        self.Cp=Cp
        self.Ca=Ca

    def server_round(self,client_updateL,alarmL):
        if True in alarmL:
            print('alarm List',alarmL)
            acc_alarm=torch.empty((2,sum(alarmL)))
            ind_alarm=0
            for i in range(len(alarmL)):
                if alarmL[i]:
                    paramt=dicadd(self.param,client_updateL[i])
                    self.test_trainer.model.load_state_dict(paramt)
                    _,acc=self.test_trainer.test()
                    acc_alarm[0,ind_alarm]=acc
                    acc_alarm[1,ind_alarm]=i
                    ind_alarm+=1
            accmax=torch.max(acc_alarm[0])
            s=torch.sum((accmax-self.acc_control)>acc_alarm[0])
            print('s',s)
            self.malious_indexL = self.malious_indexL + 1
            if s:#情况4
                print('情况4')

                count = 0
                paramdt={}
                usedL=[]
                for i in range(sum(alarmL)):
                    if acc_alarm[0,i]>accmax-self.acc_control:
                        self.malious_indexL[int(acc_alarm[1,i])]-=self.Ca+1
                        if self.malious_indexL[int(acc_alarm[1,i])]<self.Cp:
                            count += 1
                            paramdt = dicadd(paramdt, client_updateL[int(acc_alarm[1, i])])
                            usedL.append(acc_alarm[1, i].item())
                if count:
                    dicmul(paramdt, 1 / count)
                    if not self.alarm_flag:
                        self.param = dicadd(self.paraml, paramdt)
                        self.alarm_flag = True
                    else:
                        self.paraml = self.param
                        self.param = dicadd(self.param, paramdt)
                        self.alarm_flag = False
                if self.db:
                    print('------debug-------')
                    for i in range(len(alarmL)):
                        if alarmL[i]:
                            paramt = dicadd(self.paraml, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
            else:
                silent_num=len(alarmL)-sum(alarmL)
                if silent_num:
                    acc_silent = torch.empty((2, silent_num))
                    ind_alarm = 0
                    for i in range(len(alarmL)):
                        if not alarmL[i]:
                            paramt = dicadd(self.param, client_updateL[i])
                            self.test_trainer.model.load_state_dict(paramt)
                            _, acc = self.test_trainer.test()
                            acc_silent[0, ind_alarm] = acc
                            acc_silent[1, ind_alarm] = i
                            ind_alarm += 1
                    accsmax = torch.max(acc_silent[0])
                if silent_num and accsmax+self.acc_control>accmax:#情况3+假警报,认为真诚客户端均silent
                    print('情况3+假警报,认为真诚客户端均silent')
                    count=0
                    paramdt={}
                    usedL=[]
                    for i in range(len(alarmL)-sum(alarmL)):
                        if acc_silent[0,i]>accsmax-self.acc_control:
                            self.malious_indexL[int(acc_silent[1, i])] -= self.Ca + 1
                            if self.malious_indexL[int(acc_silent[1, i])] < self.Cp:
                                paramdt = dicadd(paramdt, client_updateL[int(acc_silent[1, i])])
                                count+=1
                                usedL.append(acc_silent[1, i].item())
                    if count:
                        dicmul(paramdt, 1 / count)
                        self.paraml=self.param
                        self.param = dicadd(self.param, paramdt)
                        self.alarm_flag=False

                else:#情况3+真警报,认为所有警报客户端为真诚客户端
                    print('情况3+真警报,认为所有警报客户端为真诚客户端')
                    count=0
                    paramdt={}
                    usedL=[]
                    for i in range(sum(alarmL)):
                        self.malious_indexL[int(acc_alarm[1, i])] -= self.Ca + 1
                        if self.malious_indexL[int(acc_alarm[1, i])] < self.Cp:
                            paramdt=dicadd(paramdt,client_updateL[int(acc_alarm[1,i])])
                            count+=1
                            usedL.append(acc_alarm[1,i].item())
                    if count:
                        dicmul(paramdt, 1 / count)
                        if not self.alarm_flag:
                            self.param = dicadd(self.paraml, paramdt)
                            self.alarm_flag = True
                        else:
                            self.paraml = self.param
                            self.param = dicadd(self.param, paramdt)
                            self.alarm_flag = False
                    if self.db:
                        print('------debug-------')
                        for i in range(len(alarmL)):
                            if alarmL[i]:
                                paramt = dicadd(self.paraml, client_updateL[i])
                                self.test_trainer.model.load_state_dict(paramt)
                                _, acc = self.test_trainer.test()
            print(usedL,'be added')
        else:  # 情况1和2
            print('情况1和2')
            for i in range(len(alarmL)):
                if self.malious_indexL[len(alarmL) - i - 1] > self.Cp:
                    del client_updateL[len(alarmL) - i - 1]
            if alarmL:
                super(Siren, self).server_round(client_updateL)
            print(len(alarmL), '参与了聚合')
            self.malious_indexL = self.malious_indexL - self.Ca
        self.malious_indexL = torch.nn.ReLU()(self.malious_indexL)


if __name__ == '__main__':
    print(device)
    totalL, train_size, test_size = data_to_device('MINST', device)
    X_traint, y_traint, X_testt, y_testt = totalL

    client_num = 10
    malicious_num = 4
    communication_rounds = 10
    client_epchoes = 3
    non_IID_degree=0.1

    clientL = []

    # IID 数据分配
    # ind = torch.randperm(train_size)
    # per_count = (train_size / client_num)
    # for i in range(client_num):
    #     dataL = []
    #     dataL.append(X_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
    #     dataL.append(y_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
    #     if i < malicious_num:
    #         client = mc.Sign_flipping(dataL, client_epchos=client_epchoes,
    #                                   boosting_factor=10)
    #     else:
    #         client = mc.Siren_Client(dataL, client_epchos=client_epchoes)
    #     clientL.append(client)

    # non-IID 数据分配
    indL=[torch.nonzero(y_traint==i).squeeze() for i in range(10)]
    perc_ind=torch.zeros(10)
    pmatrix=(1-non_IID_degree)/9*torch.ones(10,10)+torch.eye(10)*(non_IID_degree-(1-non_IID_degree)/9)
    indt=[]
    for i in range(10):
        indt.append(torch.cat([indL[j][int(perc_ind[j]*indL[j].shape[0]):int((perc_ind[j]+pmatrix[i,j])*indL[j].shape[0])] for j in range(10)]))
        print(i,indt[i].shape)
        perc_ind+=pmatrix[i]
    # show
    # for ten in indt:
    #     print([torch.sum(y_traint[ten]==i).item() for i in range(10)])
    for i in range(client_num):
        dataL=[]
        dataL.append(X_traint[indt[i]])
        dataL.append(y_traint[indt[i]])
        if i < malicious_num:
            client = mc.Sign_flipping(dataL, client_epchos=client_epchoes,
                                      boosting_factor=10)
        else:
            client = mc.Siren_Client(dataL, client_epchos=client_epchoes)
        clientL.append(client)


    test_trainer = Trainer(load_data='', data_L=totalL)
    initial_dict = test_trainer.model.state_dict()

    ind1 = torch.randperm(train_size)
    trainlst = [X_traint[ind1[:5000]], y_traint[ind1[:5000]]]
    # server=Server(initial_dict)
    server = Siren_PA(initial_dict, trainlst,client_num)

    for i in range(communication_rounds):
        print("round {}/{}".format(i, communication_rounds))
        client_updateL = []
        alarmL = []
        client_count = 0
        for client in clientL:
            print('---------')
            if client_count < malicious_num:
                client_update_dic = client.client_round(server.param, show_detial=False)
                attck_flag = bool(random.randint(0, 3)<1)
                # print(attck_flag)
            else:
                client_update_dic,attck_flag = client.client_round(server.param, show_detial=False)
            alarmL.append(attck_flag)
            client_updateL.append(client_update_dic)
            client_count += 1
        # print(client_updateL[0][list(server.param.keys())[0]])
        flattened_tensorL = []

        # #余弦相似度计算
        # for i in range(client_num):
        #     tens = list(client_updateL[i].values())
        #     flattened_tensor = torch.cat([t.flatten() for t in tens])
        #     flattened_tensorL.append(flattened_tensor)
        # f1 = flattened_tensorL[0]
        # f2 = sum(flattened_tensorL[1:]) / (client_num - 1)
        # cos_sim = torch.sum(f1 * f2) / (torch.sum(f1 ** 2) * torch.sum(f2 ** 2)) ** 0.5
        # print(cos_sim)

        server.server_round(client_updateL,alarmL)
        test_trainer.model.load_state_dict(server.param)
        # print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
    test_trainer.draw(0, 1)