
from FL_MINST import *


class Sign_flipping(Client):
    def __init__(self, train_dataL, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(),client_epchos=5,boosting_factor=4):
        super(Sign_flipping,self).__init__(train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F,client_epchos)
        self.sign_flipping_factor=-boosting_factor

    def client_round(self, newpara=None,show_detial=True):
        if newpara:
            self.model.load_state_dict(newpara)

        params1 = {}
        for key in self.model.state_dict().keys():
            params1[key] = self.model.state_dict()[key].clone()
        for i in range(self.client_epchos):
            self.evolve(show_d=show_detial)
        params2 = self.model.state_dict()
        params_diff = {}
        for key in params1.keys():
            params_diff[key] = (params2[key] - params1[key])*self.sign_flipping_factor
        return params_diff

class Label_flipping(Sign_flipping):
    def __init__(self, train_dataL, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(),client_epchos=5,label_flipping_method='reverse',boosting_factor=1):
        label = train_dataL[1]
        if label_flipping_method=='reverse':
            label=torch.max(label)-label
            train_dataL[1]=label
        elif label_flipping_method=='next':
            class_max=torch.max(label)
            label=label+1
            label=torch.where(label==class_max+1,0,label)
            train_dataL[1] = label
        super(Label_flipping,self).__init__(train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F,client_epchos,-boosting_factor)

class Target_poisoning(Sign_flipping):
    def __init__(self, train_dataL,attackL=None, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(), client_epchos=5, boosting_factor=1):
        if attackL==None:
            attackL=[0,1]
        label = train_dataL[1]
        label = torch.where(label == attackL[0], attackL[1], label)
        train_dataL[1] = label
        super(Target_poisoning,self).__init__(train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F,client_epchos,-boosting_factor)

class Advanced_Target_poisoning(Sign_flipping):
    def __init__(self, train_dataL,attackL=None, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(), client_epchos=5, boosting_factor=1,de_factor=1):
        self.trainer2 =Client(train_dataL,client_epchos=client_epchos)
        self.de_factor=de_factor
        if attackL==None:
            attackL=[0,1]
        label = train_dataL[1]
        label = torch.where(label == attackL[0], attackL[1], label)
        train_dataL[1] = label
        super(Advanced_Target_poisoning,self).__init__(train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F,client_epchos,-boosting_factor)

    def client_round(self, newpara=None,show_detial=True):
        params_diff=super().client_round(newpara,show_detial)
        params_diff1=self.trainer2.client_round(newpara,False)
        for key in params_diff.keys():
            params_diff[key]=params_diff[key]-self.de_factor*params_diff1[key]
        return params_diff

class Siren_Client(Client):
    def __init__(self, train_dataL, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(), client_epchos=5, acc_control=4,re_use=True,split_ratio=0.7):
        if re_use:
            dataLi = [train_dataL[0], train_dataL[1], train_dataL[0], train_dataL[1]]
        else:
            X_traint,y_traint=train_dataL
            ind=torch.randperm(y_traint.shape[0])
            split_ind=int(y_traint.shape[0]*split_ratio)
            X_testt=X_traint[ind[split_ind:]]
            y_testt=y_traint[ind[split_ind:]]
            X_traint=X_traint[ind[:split_ind]]
            y_traint=y_traint[ind[:split_ind]]
            dataLi=[X_traint,y_traint,X_testt,y_testt]
            train_dataL=[X_traint,y_traint]

        super(Siren_Client, self).__init__(train_dataL, batch_size, bacth_size1, optimizer, para_dict, loss_F,
                                            client_epchos)

        self.test_trainer=Trainer('',data_L=dataLi)
        self.acclast=0
        self.acc_control=acc_control
        self.client_epchos=client_epchos

    # def client_round(self,newpara=None,show_detial=False,newdataL=None):
    #     if newdataL:
    #         X_testt=newdataL[0]
    #         y_testt=newdataL[1]
    #         test_size=y_testt.shape[0]
    #     else:
    #         X_testt = self.X_traint
    #         y_testt = self.y_traint
    #         test_size=self.train_size
    #     if newpara:
    #         self.testmodel.load_state_dict(newpara)
    #         model=self.testmodel
    #     else:
    #         model=self.model
    #
    #
    #     model.eval()
    #     testing_loss = 0.0
    #     testing_correct = 0.0
    #     for a, b in BatchIndex(test_size, self.batch_size, False):
    #         X_test, y_test = X_testt[a:b], y_testt[a:b]
    #         outputs = model(X_test)
    #         loss = self.loss_F(outputs, y_test)
    #         testing_loss += loss.item() * (b - a)
    #         _, pred = torch.max(outputs, 1)
    #         testing_correct += torch.sum(torch.eq(pred, y_test))
    #
    #     acc=testing_correct/test_size
    #     attack_flag=(acc-self.acclast)<self.acc_control
    #
    #     if not attack_flag:
    #         self.model,self.testmodel=self.testmodel,self.model
    #
    #     params1 = dict(self.model.state_dict())
    #     for i in range(self.client_epchos):
    #         self.evolve(show_d=show_detial)
    #
    #     params2=self.model.state_dict()
    #     params_diff = {}
    #     for key in params1.keys():
    #         params_diff[key] = params2[key] - params1[key]
    #     return params_diff,attack_flag

    def client_round(self, newpara=None,show_detial=True):

        _,accc=self.test_trainer.test(show_detail=show_detial)
        self.test_trainer.model.load_state_dict(newpara)

        _,accs=self.test_trainer.test(show_detail=show_detial)
        #print('accc:',accc,'accs',accs)
        if accs>accc-self.acc_control:
            attack_flag=False
            param_diff=super().client_round(newpara,show_detial)
        else:
            attack_flag=True
            #print('Alarming!')
            for i in range(self.client_epchos):
                self.evolve(show_d=show_detial)
            params2 = self.model.state_dict()
            param_diff=dicadd(params2,newpara,-1)
        self.test_trainer.model.load_state_dict(self.model.state_dict())
        return param_diff,attack_flag



if __name__ == '__main__':
    t=time.time()

    print(device)
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
    malicious_num=0
    communication_rounds=10
    client_epchoes=1

    ind=torch.randperm(train_size)
    per_count=(train_size/client_num)
    clientL=[]
    for i in range(client_num):
        dataL=[]
        dataL.append(X_traint[ind[int(per_count*i):int(per_count*(i+1))]])
        dataL.append(y_traint[ind[int(per_count*i):int(per_count*(i+1))]])
        if i<malicious_num:
            client=Sign_flipping(dataL,client_epchos=client_epchoes,boosting_factor=-1)
        else:
            client=Client(dataL,client_epchos=client_epchoes)
        clientL.append(client)

    test_trainer=Trainer(load_data='',data_L=totalL)

    initial_dict=test_trainer.model.state_dict()
    server=Server(initial_dict)
    for i in range(communication_rounds):
        print("round {}/{}".format(i,communication_rounds))
        client_updateL=[]
        for client in clientL:
            #print('---------')
            client_update_dic=client.client_round(server.param,show_detial=False)
            client_updateL.append(client_update_dic)
        #print(client_updateL[0][list(server.param.keys())[0]])

        # flattened_tensorL=[]
        #
        # for i in range(client_num):
        #     tens=list(client_updateL[i].values())
        #     flattened_tensor = torch.cat([t.flatten() for t in tens])
        #     flattened_tensorL.append(flattened_tensor)
        # f1=flattened_tensorL[0]
        # f2=sum(flattened_tensorL[1:])/(client_num-1)
        # cos_sim=torch.sum(f1*f2)/(torch.sum(f1**2)*torch.sum(f2**2))**0.5
        # print(cos_sim)

        server.server_round(client_updateL)
        test_trainer.model.load_state_dict(server.param)
        #print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
    #test_trainer.draw(0, 1, save_loss='sign_flipping_loss.png', save_accuracy='sign_flipping_accruacy.png')
    print(time.time()-t)