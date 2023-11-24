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

def exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,
        defence=fs.Siren_PA,defence_dic=None,IID=False):

    clientL = []
    if IID:
        ind = torch.randperm(train_size)
        per_count = (train_size / client_num)
        for i in range(client_num):
            dataL = []
            dataL.append(X_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            dataL.append(y_traint[ind[int(per_count * i):int(per_count * (i + 1))]])
            if i < malicious_num:
                client = fc.Target_poisoning(dataL, client_epchos=client_epchoes)
            else:
                client = fc.Siren_Client(dataL, client_epchos=client_epchoes,acc_control=Cc)
            clientL.append(client)

    else:
        indL = [torch.nonzero(y_traint == i).squeeze() for i in range(10)]
        perc_ind = torch.zeros(10)
        pmatrix = (1 - non_IID_degree) / 9 * torch.ones(10, 10) + torch.eye(10) * (
                non_IID_degree - (1 - non_IID_degree) / 9)
        indt = []
        data_c=[[],[]]
        for i in range(10):
            indt.append(torch.cat(
                [indL[j][int(perc_ind[j] * indL[j].shape[0]):int((perc_ind[j] + pmatrix[i, j]) * indL[j].shape[0])] for
                 j in
                 range(10)]))
            print(i, indt[i].shape)
            incid = torch.randperm(indt[i].shape[0])
            indt[i] = indt[i][incid]
            perc_ind += pmatrix[i]
        # show
        # for ten in indt:
        #     print([torch.sum(y_traint[ten]==i).item() for i in range(10)])
        pind = int(client_num / 10)
        for i in range(10):
            indtotal = indt[i]
            isize = indtotal.shape[0]
            for j in range(pind):
                dataL = []
                dataL.append(X_traint[indtotal[int(isize / pind * j):int(isize / pind * (j + 1))]])
                dataL.append(y_traint[indtotal[int(isize / pind * j):int(isize / pind * (j + 1))]])
                # print(dataL[1].shape)
                if i < int(10 * malicious_num / client_num):
                    data_c[0].append(X_traint[indtotal[int(isize / pind * j):int(isize / pind * (j + 1))]])
                    data_c[1].append(y_traint[indtotal[int(isize / pind * j):int(isize / pind * (j + 1))]])
                    client=None
                else:
                    if defence in defenceL[:4]:
                        client = Client(dataL, client_epchos=client_epchoes, batch_size=1024)
                    else:
                        client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc, batch_size=1024)
                clientL.append(client)

        dataX=torch.concat(data_c[0])
        datay=torch.concat(data_c[1])
        mal_client=fc.Target_poisoning([dataX,datay], client_epchos=client_epchoes, batch_size=1024,mali_p=malicious_num/client_num)


    test_trainer = Trainer(load_data='', data_L=totalL,bacth_size1=4096)
    targetmodel_trainerL=[]
    for i in range(10):
        targetmodel_test=Trainer(load_data='',data_L=tmLL[i])
        targetmodel_trainerL.append(targetmodel_test)


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
        fake_update=mal_client.client_round(server.param,show_detial=False)
        for client in clientL:
            #print('---------')
            if client_count < malicious_num:
                client_update_dic=fake_update
                attck_flag = bool(random.randint(0, 3) < 1)
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
        print('-------------------')
        for t in targetmodel_trainerL:
            t.model.load_state_dict(server.param)
            t.test()
    return test_trainer.test_accuracyL,[t.test_accuracyL for t in targetmodel_trainerL],server



if __name__ == '__main__':

    Cs = args.Cs
    Cc = args.Cc
    Cp = args.Cp
    Ca = args.Ca
    rootsize = args.rs

    client_num = args.cn
    malicious_numL = args.mn
    communication_rounds = args.cr
    client_epchoes = args.ce
    non_IID_degreeL = args.nid
    use_IID = args.id

    defence_indl = args.defence
    for i in range(len(defence_indl)):
        assert 0 <= defence_indl[i] <= 6, "elements of -defence should in 0~6"

    data_set = args.dataset

    print("running in the device {}".format(device))

    totalL, train_size, test_size = data_to_device_s(data_set, device)
    X_traint, y_traint, X_testt, y_testt = totalL
    tmLL=[]
    tsize=[]
    for i in range(10):
        testind = (y_testt == i)
        testy = y_testt[testind]
        testx = X_testt[testind]
        print('testszie', testx.shape, testy.shape)
        tsize.append(testy.shape[0])
        tmL = [X_traint, X_testt, testx, testy]
        tmLL.append(tmL)


    defenceL = [ expServer,expKrum,expCoord, expFLTrust, fs.Siren, fs.Siren_fltrust, fs.Siren_PA]
    defence_dicL = [{}, {'client_num': client_num, 'malicious_client_num': malicious_numL[0]}, {},{},
                    {'acc_control': Cs}, {'acc_control': Cs}, None]
    defenceL_name = ['Server','Krum',  'Coord','FLTrust', 'Siren', 'Siren_fl', 'Siren_PA']
    ti=time.time()
    total_dic={}
    target_dic={}
    pa_data={}
    with open('tlog.txt', 'w+', encoding='utf-8') as f:
        pass
    for malicious_num in malicious_numL:
        for non_IID_degree in non_IID_degreeL:
            for i in defence_indl:
                df = defenceL[i]
                defence_dicL[1]['malicious_client_num'] = malicious_num
                dfd = defence_dicL[i]
                l1,l2,ser=exp(client_num,malicious_num,communication_rounds,client_epchoes,non_IID_degree,df,dfd)
                ctime=time.time()-ti
                ti=time.time()
                idn=str(defenceL_name[i])+'\tmalicious_num:'+str(malicious_num)+'\tnon_IID_degree:'+str(non_IID_degree)
                tt=0
                for k in range(10):
                    tt+=tsize[k]*l2[k][-1]
                tt/=10000
                with open('tlog.txt','a',encoding='utf-8') as f:
                    f.write("{:.3f}\t".format(l1[-1])+"{:.3f}\t".format(tt)+'\tcost:{:.3f} '.format(ctime)+idn+'\n')
                    for k in range(10):
                        f.write("{:.3f}".format(l2[k][-1])+'\t')
                    f.write('\n')
                total_dic[idn]=torch.tensor(l1)
                target_dic[idn]=torch.tensor(l2)
                if i == 6:
                    pa_data[idn] = ser.malious_indexL
                print(total_dic[idn],target_dic[idn])
    # 'arres.pth' is a dict. each value is a tensor,it records the total accuracy in each communication rounds with the condition of its key.
    torch.save(total_dic,'P_total'+str(client_num)+'.pth')
    # 'arres.pth' is a dict. each value is a tensor,it records the accuracy of target class in each communication rounds with the condition of its key.
    torch.save(target_dic,'P_target'+str(client_num)+'.pth')
    # 'pa_data.pth' is a dict. It records the PA value of each client if you use server 'Siren_PA'
    torch.save(pa_data, 'P_pa_data'+str(client_num)+'.pth')