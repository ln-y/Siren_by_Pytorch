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

def exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,Attack=fc.Sign_flipping,
        defence=fs.Siren_PA,defence_dic=None,IID=False):
    print(int(malicious_num/client_num*10))
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
            incid=torch.randperm(indt[i].shape[0])
            indt[i]=indt[i][incid]
            perc_ind += pmatrix[i]
        # show
        # for ten in indt:
        #     print([torch.sum(y_traint[ten]==i).item() for i in range(10)])
        pind=int(client_num/10)
        for i in range(10):
            indtotal=indt[i]
            isize=indtotal.shape[0]
            for j in range(pind):
                dataL = []
                dataL.append(X_traint[indtotal[int(isize/pind*j):int(isize/pind*(j+1))]])
                dataL.append(y_traint[indtotal[int(isize/pind*j):int(isize/pind*(j+1))]])
                #print(dataL[1].shape)
                if i < int(10*malicious_num/client_num):
                    client = Attack(dataL, client_epchos=client_epchoes,
                                    boosting_factor=10, batch_size=1024)
                else:
                    if defence in defenceL[:3]:
                        client = Client(dataL, client_epchos=client_epchoes, batch_size=1024)
                    else:
                        client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc, batch_size=1024)
                clientL.append(client)



    test_trainer = Trainer(load_data='', data_L=totalL,batch_size=1024,bacth_size1=4096)
    #test_trainer.evolve()
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
                attck_flag = bool(random.random() < 0.1)
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
    return test_trainer.test_accuracyL,server



if __name__ == '__main__':
    Cs=10
    Cc=4
    Cp=4.5
    Ca=0.5
    rootsize=100

    client_num = 200
    malicious_num = 1
    communication_rounds =12
    client_epchoes = 1
    non_IID_degree = 0.1

    attackL = [fc.Sign_flipping, fc.Label_flipping, fc.Target_poisoning]
    defenceL = [expServer, expKrum, expCoord, fs.Siren, fs.Siren_fltrust, fs.Siren_PA]
    defence_dicL=[{}, {'client_num': client_num,'malicious_client_num':malicious_num},{},
                  {'acc_control': Cs},{'acc_control': Cs},None]
    attackL_name=['Sign_flipping', 'Label_flipping', 'Target_poisoning']
    defenceL_name=['Server', 'Krum', 'Coord', 'Siren', 'Siren_fl', 'Siren_PA']
    print(device)

    totalL, train_size, test_size = data_to_device_s('MINST', device)
    X_traint, y_traint, X_testt, y_testt = totalL

    with open('mlog50.txt', 'w+', encoding='utf-8') as f:
        pass
    dic_data={}
    pa_data={}
    ti=time.time()
    for j in range(len(attackL)):
        attack=attackL[j]
        for i in range(len(defenceL)):
            for non_IID_degree in [0.1,0.5]:
                for mal_p in [0.4,0.8]:
                    defence = defenceL[i]
                    defence_dic = defence_dicL[i]
                    expLabel = attackL_name[j] + '+' + defenceL_name[i] + '+mal_p'+ str(mal_p) + ' IID=' + str(non_IID_degree)
                    print('---------------------' + expLabel + '---------------------')
                    accL, ser = exp(client_num, int(mal_p*client_num), communication_rounds, client_epchoes, non_IID_degree, attack,
                                    defence,defence_dic)
                    acct = torch.tensor(accL)
                    dic_data[expLabel] = acct
                    ctime = time.time() - ti
                    ti = time.time()
                    with open('mlog200.txt', 'a', encoding='utf-8') as f:
                        f.write(str(j + 1) + ' ' + str(i + 1) + ' ' + "{:.3f}".format(
                            accL[-1]) + '\t' + expLabel + '\tcost:' + "{:.3f}".format(ctime) + '\n')
                    if i == 5:
                        pa_data[expLabel] = ser.malious_indexL

    torch.save(dic_data,'accres200.pth')
    torch.save(pa_data,'pa_data200.pth')






