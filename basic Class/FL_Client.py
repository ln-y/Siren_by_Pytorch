import torch
import torch.nn as nn
import time
from FL import Client,Server,Trainer,myLoadi,myLoadt,CNN,BatchIndex

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
