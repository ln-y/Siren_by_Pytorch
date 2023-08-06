import time

from FL_MINST import *
import random
import FL_Server_defence0 as fs
import FL_Client as fc

class expServer(Server):
    def __init__(self,pretrained_data,root_train_data,):
        super(expServer,self).__init__(pretrained_data)

    def server_round(self,client_updateL,alarmL):
        super().server_round(client_updateL)

class expKrum(fs.Krum):
    def __init__(self,pretrained_data,root_train_data,client_num,malicious_client_num):
        super().__init__(pretrained_data,client_num,malicious_client_num)

    def server_round(self,client_updateL,alarmL):
        super().server_round(client_updateL)

class expCoord(fs.Coordinate_wise_Median):
    def __init__(self, pretrained_data, root_train_data):
        super().__init__(pretrained_data)

    def server_round(self, client_updateL, alarmL):
        super().server_round(client_updateL)

def exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,
        defence=fs.Siren_PA,defence_dic=None,IID=False,Attack=fc.Target_poisoning,):

    clientL = []
    if IID:
        ind = torch.randperm(train_size)
        per_count = (train_size / client_num)
        for i in range(client_num):
            dataL = []
            dataL.append(X_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            dataL.append(y_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            if i < malicious_num:
                client = Attack(dataL, client_epchos=client_epchoes,
                                          boosting_factor=10)
            else:
                client = fc.Siren_Client(dataL, client_epchos=client_epchoes,acc_control=Cc)
            clientL.append(client)

    else:
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
                client = Attack(dataL, client_epchos=client_epchoes,
                                          boosting_factor=10,batch_size=1024)
            else:
                if defence in defenceL[:3]:
                    client=Client(dataL,client_epchos=client_epchoes,batch_size=1024)
                else:
                    client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc,batch_size=1024)
            clientL.append(client)



    test_trainer = Trainer(load_data='', data_L=totalL,bacth_size1=4096)
    tagetmodel_test=Trainer(load_data='',data_L=tmL)
    initial_dict = test_trainer.model.state_dict()

    ind1 = torch.randperm(train_size)
    trainlst = [X_traint[ind1[:rootsize]], y_traint[ind1[:rootsize]]]

    if defence_dic==None:
        defence_dic={'client_num': client_num,
                     'acc_control': Cs,
                   'Cp': Cp,
                   'Ca': Ca}
    ref_dic = {'pretrained_data': initial_dict,
                   'root_train_data': trainlst,
                   }
    # server=Server(initial_dict)
    ref_dic.update(defence_dic)
    server = defence(**ref_dic)


    for i in range(communication_rounds):
        print("round {}/{}".format(i, communication_rounds))
        client_updateL = []
        alarmL = []
        client_count = 0
        for client in clientL:
            #print('---------')
            if client_count < malicious_num:
                client_update_dic = client.client_round(server.param, show_detial=False)
                attck_flag = bool(random.randint(0, 3) < 1)
                # print(attck_flag)
            else:
                if defence in defenceL[:3]:
                    client_update_dic=client.client_round(server.param,show_detial=False)
                    attck_flag=False
                else:
                    client_update_dic, attck_flag = client.client_round(server.param, show_detial=False)
            alarmL.append(attck_flag)
            client_updateL.append(client_update_dic)
            client_count += 1

        server.server_round(client_updateL, alarmL)
        test_trainer.model.load_state_dict(server.param)
        # print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
        tagetmodel_test.test()
    return test_trainer.test_accuracyL,tagetmodel_test.test_accuracyL



if __name__ == '__main__':
    Cs=10
    Cc=4
    Cp=4.5
    Ca=0.5
    rootsize=100

    client_num = 10
    malicious_num = 1
    communication_rounds = 3
    client_epchoes = 1
    non_IID_degree = 0.1

    print(device)

    totalL, train_size, test_size = data_to_device_s('MINST', device)
    X_traint, y_traint, X_testt, y_testt = totalL
    testind = (y_testt == 0)
    testy = y_testt[testind]
    testx = X_testt[testind]
    print('testszie', testx.shape, testy.shape)
    tmL = [X_traint, X_testt, testx, testy]

    defenceL = [expServer, expKrum, expCoord, fs.Siren, fs.Siren_fltrust, fs.Siren_PA]
    defence_dicL = [{}, {'client_num': client_num, 'malicious_client_num': malicious_num}, {},
                    {'acc_control': Cs}, {'acc_control': Cs}, None]
    defenceL_name = ['Server', 'Krum', 'Coord', 'Siren', 'Siren_fl', 'Siren_PA']
    ti=time.time()
    total_dic={}
    target_dic={}
    for malicious_num in [4,8]:
        for non_IID_degree in [0.1,0.5]:
            for i in range(len(defenceL)):
                df=defenceL[i]
                dfd=defence_dicL[i]
                l1,l2=exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,df,dfd)
                ctime=time.time()-ti
                ti=time.time()
                idn=str(defenceL_name[i])+'\tmalicious_num:'+str(malicious_num)+'\tnon_IIS_degree:'+str(non_IID_degree)
                with open('tlog.txt','a',encoding='utf-8') as f:
                    f.write(idn+'\t'+"{:.3f}\t".format(l1[-1])+"{:.3f}".format(l2[-1])+'\tcost:{:.3f}\n'.format(ctime))
                total_dic[idn]=torch.tensor(l1)
                target_dic[idn]=torch.tensor(l2)
    torch.save(total_dic,'P_total.pth')
    torch.save(target_dic,'P_target.pth')