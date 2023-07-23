import random
import torch
from FL import Client,Server,Trainer,data_to_device,dicadd,dicmul
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
