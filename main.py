import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-Cs', type=float, default=10)
parser.add_argument('-Cc', type=float, default=4)
parser.add_argument('-Cp', type=float, default=4.5)
parser.add_argument('-Ca', type=float, default=0.5)
parser.add_argument('-rs', type=int, help='rootsize', default=100)
parser.add_argument('-cn', type=int, help='client num', default=10)
parser.add_argument('-mn', type=int, nargs='+', help='malicious num', default=[4, 8])
parser.add_argument('-cr', type=int, help='communication rounds', default=40)
parser.add_argument('-ce', type=int, help='client epchoes', default=5)
parser.add_argument('-nid', type=float, nargs='+', help='non_IID_degree', default=[0.1, 0.5])
parser.add_argument('--id', action='store_true',
                    help='It will let dataset distributed in IID way,and let --nid have no effect')
parser.add_argument('-attack', type=int, nargs='+',
                    help='the type of attack method(0 is no attack, 1 is Sign_flipping, 2 is Label_flipping',
                    default=[0, 1, 2])
parser.add_argument('-defence', type=int, nargs='+',
                    help='the type of defence method(0 is no defence, 1 is Krum, 2 is Coordinate wise Median, 3 is FLTrust, 4 is Siren , 5 is Siren_fl , 6 is Siren_PA',
                    default=[0, 1, 2, 3, 4, 5, 6])
parser.add_argument("-dataset", type=str, help='The type of dataset', default='MNIST', choices=['MNIST', 'CIFAR10'])
parser.add_argument("-device",type=str,help='the device to simulate',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
print(args)
with open('config.py','w') as f:
    f.write("device='"+args.device+"'")

from FL_Base import *
import random
import FL_Server_defence_ as fs
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

class expFLTrust(fs.FLTrust):
    def __init__(self, pretrained_data, root_train_data):
        super().__init__(pretrained_data,root_train_data)

    def server_round(self, client_updateL, alarmL):
        super().server_round(client_updateL)

def exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,Attack=fc.Sign_flipping,
        defence=fs.Siren_PA,defence_dic=None,IID=False):

    clientL = []
    if IID:
        ind = torch.randperm(train_size)
        per_count = (train_size / client_num)
        for i in range(client_num):
            dataL = []
            dataL.append(X_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            dataL.append(y_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            if i < malicious_num and Attack!=fc.Client:
                client = Attack(dataL, client_epchos=client_epchoes,
                                          boosting_factor=10)
            else:
                if defence in defenceL[:4]:
                    client = Client(dataL, client_epchos=client_epchoes, batch_size=1024)
                else:
                    client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc, batch_size=1024)
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
            if i < malicious_num and Attack!=fc.Client:
                client = Attack(dataL, client_epchos=client_epchoes,
                                          boosting_factor=10,batch_size=1024)
            else:
                if defence in defenceL[:4]:
                    client=Client(dataL,client_epchos=client_epchoes,batch_size=1024)
                else:
                    client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc,batch_size=1024)
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
            if client_count < malicious_num and Attack!=fc.Client:
                client_update_dic = client.client_round(server.param, show_detial=False)
                attck_flag = bool(random.random() < 0.1)
                # print(attck_flag)
            else:
                if defence in defenceL[:4]:
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

    Cs=args.Cs
    Cc=args.Cc
    Cp=args.Cp
    Ca=args.Ca
    rootsize=args.rs

    client_num = args.cn
    malicious_numL = args.mn
    communication_rounds = args.cr
    client_epchoes = args.ce
    non_IID_degreeL = args.nid
    use_IID=args.id

    attack_indl=args.attack
    defence_indl=args.defence
    for i in range(len(attack_indl)):
        assert 0<= attack_indl[i] <=2, "elements of -attack should in 0~2"
    for i in range(len(defence_indl)):
        assert 0<= defence_indl[i] <=6, "elements of -defence should in 0~6"

    data_set=args.dataset

    attackL = [fc.Client, fc.Sign_flipping, fc.Label_flipping]
    defenceL = [expServer, expKrum, expCoord,expFLTrust, fs.Siren, fs.Siren_fltrust, fs.Siren_PA]
    defence_dicL=[{}, {'client_num': client_num,'malicious_client_num':malicious_numL[0]},{},{},
                  {'acc_control': Cs},{'acc_control': Cs},None]
    attackL_name=['No attack','Sign_flipping', 'Label_flipping']
    defenceL_name=['Server', 'Krum', 'Coord', 'FLTrust', 'Siren', 'Siren_fl', 'Siren_PA']
    print("running in the device {}".format(device))
    # attackL1=[attackL[i] for i in attack_indl]
    # defenceL1=[defenceL[i] for i in defence_indl]
    # attackL_name1 = [attackL_name[i] for i in attack_indl]
    # defenceL_name1 = [defenceL_name[i] for i in defence_indl]


    totalL, train_size, test_size = data_to_device_s(args.dataset, device)
    X_traint, y_traint, X_testt, y_testt = totalL

    with open('mlog.txt', 'w+', encoding='utf-8') as f:
        pass
    dic_data={}
    pa_data={}
    ti=time.time()
    for j in attack_indl:
        attack=attackL[j]
        for i in defence_indl:
            for non_IID_degree in non_IID_degreeL:
                for malicious_num in malicious_numL:
                    defence = defenceL[i]
                    defence_dicL[1]['malicious_client_num']=malicious_num
                    defence_dic = defence_dicL[i]
                    accL, ser = exp(client_num, malicious_num, communication_rounds, client_epchoes, non_IID_degree, attack,
                                    defence,
                                    defence_dic, IID=use_IID)
                    expLabel = attackL_name[j] + '+' + defenceL_name[i] + '+mal_num_{}'.format(malicious_num) + ' IID=' + str(non_IID_degree)
                    acct = torch.tensor(accL)
                    dic_data[expLabel] = acct
                    ctime = time.time() - ti
                    ti = time.time()
                    with open('mlog.txt', 'a', encoding='utf-8') as f:
                        f.write(str(j + 1) + ' ' + str(i + 1) + ' ' + "{:.3f}".format(
                            accL[-1]) + '\t' + expLabel + '\tcost:' + "{:.3f}".format(ctime) + '\n')
                    if i == 6:
                        pa_data[expLabel] = ser.malious_indexL
    #'arres.pth' is a dict. each value is a tensor,it records the accuracy in each communication rounds with the condition of its key.
    torch.save(dic_data,'accres.pth')
    #'pa_data.pth' is a dict. It records the PA value of each client if you use server 'Siren_PA'
    torch.save(pa_data,'pa_data.pth')






