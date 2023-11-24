
from FL_Base import *
import FL_Server_defence_ as fs

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

class Target_poisoning(Client):
    def __init__(self, train_dataL,attackL=None, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(), client_epchos=5, mali_p=0.1,pretrain_factor=1):
        if attackL==None:
            attackL=[8,9]

        self.test_trainer = Client(train_dataL, batch_size, bacth_size1, optimizer, para_dict, loss_F,client_epchos=client_epchos)
        label = train_dataL[1].clone()
        ind0 = (label == attackL[0])
        target_X = train_dataL[0][ind0]
        target_y = train_dataL[1][ind0]
        self.evaluater = Client([target_X, target_y], batch_size, bacth_size1, optimizer, para_dict, loss_F)
        label[ind0]=attackL[1]
        train_dataL[1] = label
        super(Target_poisoning,self).__init__(train_dataL,batch_size,bacth_size1,optimizer,para_dict,loss_F,
                                              client_epchos)
        for i in range(pretrain_factor*client_epchos):
            self.evolve(show_d=False)
        _,self.acc_val=self.test(show_detail=False)
        print('acc_Val',self.acc_val)
        self.client_round=self.client_round1
        self.mali_p=mali_p
        self.attack_flag=False

    def client_round1(self, newpara=None,show_detial=True):

        if newpara:
            self.test_trainer.model.load_state_dict(newpara)
        if self.attack_flag:
            print('begin attack')

            for i in range(self.client_epchos):
                self.evolve(show_d=show_detial)
                self.test_trainer.evolve(show_d=show_detial)

            guess_para = self.test_trainer.model.state_dict()
            dst_para = dicpoly(guess_para, newpara, 1 - self.mali_p, self.mali_p)
            return dicpoly(self.model.state_dict(), dst_para,
                           1 / self.mali_p, -1 / self.mali_p)
        else:
            _, acc = self.test_trainer.test(show_detail=False)
            print(acc)
            if acc >= self.acc_val:
                self.attack_flag=True
            print('no attack')
            return self.test_trainer.client_round(newpara=newpara, show_detial=show_detial)

    def client_round0(self, newpara=None,show_detial=True):
        if newpara:
            self.test_trainer.model.load_state_dict(newpara)
        _,acc=self.test_trainer.test(show_detail=False)
        print(acc)
        if acc>=self.acc_val:
            print('begin attack')
            self.evaluater.model.load_state_dict(newpara)
            _,accx=self.evaluater.test(show_detail=False)
            print('now accx:',accx)
            if accx>0.5*acc:
                self.greater_model=tensor_dic_copy(newpara)
                print('get Great model')

            for i in range(self.client_epchos):
                self.evolve(show_d=show_detial)

            return dicpoly(self.model.state_dict(),self.greater_model,
                           -1/self.mali_p,-1/self.mali_p)
        else:
            print('no attack')
            return self.test_trainer.client_round(newpara=newpara,show_detial=show_detial)


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
        self.model.load_state_dict(newpara)
        self.test(show_detail=False)
        qacc=self.test_accuracyL[-1]
        print('qacc',qacc)
        if qacc>60:
            params_diff = super().client_round(newpara, show_detial)
            params_diff1 = self.trainer2.client_round(newpara, False)
            for key in params_diff.keys():
                params_diff[key] = params_diff[key] - self.de_factor * params_diff1[key]
            return params_diff
        else:
            params_diff1 = self.trainer2.client_round(newpara, False)
            return params_diff1


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
        if show_detial:
            print('-----Siren_client test--------')
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

class Target_model_poisoning(Client):
    def __init__(self,train_dataL,attackL=None, batch_size=64, bacth_size1=512, optimizer=torch.optim.Adam, para_dict=None,
                 loss_F=nn.CrossEntropyLoss(), client_epchos=5):
        if attackL == None:
            attackL = [5, 7]
        label = train_dataL[1]
        label = torch.where(label == attackL[0], attackL[1], label)
        train_dataL[1] = label
        super(Target_model_poisoning, self).__init__(train_dataL, batch_size, bacth_size1, optimizer, para_dict,
                                                        loss_F, client_epchos)





if __name__ == '__main__':
    t=time.time()

    print(device)

    totalL,train_size,test_size=data_to_device_s('MINST',device)
    X_traint, y_traint, X_testt, y_testt = totalL

    client_num=10
    malicious_num=4
    communication_rounds=40
    client_epchoes=5
    non_IID_degree=0.5

    tmLL = []
    tsize = []
    for i in range(10):
        testind = (y_testt == i)
        testy = y_testt[testind]
        testx = X_testt[testind]
        print('testszie', testx.shape, testy.shape)
        tsize.append(testy.shape[0])
        tmL = [X_traint, X_testt, testx, testy]
        tmLL.append(tmL)
    test_trainerL=[]
    for i in range(10):
        test_trainera=Trainer(load_data='',data_L=tmLL[i])
        test_trainerL.append(test_trainera)

    ind=torch.randperm(train_size)
    per_count=(train_size/client_num)
    clientL=[]
    indL = [torch.nonzero(y_traint == i).squeeze() for i in range(10)]
    perc_ind = torch.zeros(10)
    pmatrix = (1 - non_IID_degree) / 9 * torch.ones(10, 10) + torch.eye(10) * (
            non_IID_degree - (1 - non_IID_degree) / 9)
    indt = []
    for i in range(10):
        indt.append(torch.cat(
            [indL[j][int(perc_ind[j] * indL[j].shape[0]):int((perc_ind[j] + pmatrix[i, j]) * indL[j].shape[0])] for
             j in
             range(10)]))
        print(i, indt[i].shape)
        perc_ind += pmatrix[i]
    # show
    # for ten in indt:
    #     print([torch.sum(y_traint[ten]==i).item() for i in range(10)])
    for i in range(client_num):
        dataL = []
        dataL.append(X_traint[indt[i]])
        dataL.append(y_traint[indt[i]])
        if i < malicious_num:
            client = Target_poisoning(dataL, client_epchos=client_epchoes,
                            mali_p=malicious_num/client_num, batch_size=1024)
        else:
            client = Client(dataL, client_epchos=client_epchoes, batch_size=1024)
        clientL.append(client)

    test_trainer=Trainer(load_data='',data_L=totalL)

    initial_dict=test_trainer.model.state_dict()
    server=fs.Coordinate_wise_Median(initial_dict)
    mal_server=Server(initial_dict)

    for i in range(communication_rounds):
        print("round {}/{}".format(i,communication_rounds))
        client_updateL=[]
        mal_updateL=[]
        for i in range(len(clientL)):
            client = clientL[i]
            if i<malicious_num:
                client_update_dic,mal_update = client.client_round(server.param, show_detial=False,m_newpara=mal_server.param)
                client_updateL.append(client_update_dic)
                mal_updateL.append(mal_update)
            else:

                # print('---------')
                client_update_dic = client.client_round(server.param, show_detial=False)
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
        mal_server.server_round(mal_updateL)
        test_trainer.model.load_state_dict(server.param)
        #print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
        for tt in test_trainerL:
            tt.model.load_state_dict(server.param)
            tt.test(show_detail=False)
            print("{:.3f}".format(tt.test_accuracyL[-1]),end=' ')
        print('\nattacker model:')
        test_trainer.model.load_state_dict(mal_server.param)
        test_trainer.test()
        for tt in test_trainerL:
            tt.model.load_state_dict(mal_server.param)
            tt.test(show_detail=False)
            print("{:.3f}".format(tt.test_accuracyL[-1]),end=' ')

        print()
    #test_trainer.draw(0, 1, save_loss='sign_flipping_loss.png', save_accuracy='sign_flipping_accruacy.png')
    print(time.time()-t)