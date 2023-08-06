import time

from FL_MINST import *
import random
import FL_Server_defence as fs
import FL_Client as fc

def exp(Cs,Cc,Cp,Ca,rootsize,attack=fc.Sign_flipping):
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
    indL = [torch.nonzero(y_traint == i).squeeze() for i in range(10)]
    perc_ind = torch.zeros(10)
    pmatrix = (1 - non_IID_degree) / 9 * torch.ones(10, 10) + torch.eye(10) * (
            non_IID_degree - (1 - non_IID_degree) / 9)
    indt = []
    for i in range(10):
        indt.append(torch.cat(
            [indL[j][int(perc_ind[j] * indL[j].shape[0]):int((perc_ind[j] + pmatrix[i, j]) * indL[j].shape[0])] for j in
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
            client = attack(dataL, client_epchos=client_epchoes,batch_size=1024,bacth_size1=8192,
                                      boosting_factor=10)
        else:
            client = fc.Siren_Client(dataL, client_epchos=client_epchoes, acc_control=Cc,batch_size=1024,bacth_size1=8192)
        clientL.append(client)

    test_trainer = Trainer(load_data='', data_L=totalL,bacth_size1=8192)
    initial_dict = test_trainer.model.state_dict()

    ind1 = torch.randperm(train_size)
    trainlst = [X_traint[ind1[:rootsize]], y_traint[ind1[:rootsize]]]
    # server=Server(initial_dict)
    server = fs.Siren_PA(initial_dict, trainlst, 10, Cs, Cp, Ca)

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
                client_update_dic, attck_flag = client.client_round(server.param, show_detial=False)
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

        server.server_round(client_updateL, alarmL)
        test_trainer.model.load_state_dict(server.param)
        # print(server.param[list(server.param.keys())[0]])
        test_trainer.test()
    return max(test_trainer.test_accuracyL[-1],test_trainer.test_accuracyL[-2])

if __name__ == '__main__':
    Cs=10
    Cc=4
    Cp=4.5
    Ca=0.5
    rootsize=100


    print(device)

    totalL, train_size, test_size = data_to_device('MINST', device)
    X_traint, y_traint, X_testt, y_testt = totalL

    client_num = 10
    malicious_num = 4
    communication_rounds = 15
    client_epchoes = 1
    non_IID_degree = 0.1
    attackL=[fc.Sign_flipping,fc.Label_flipping,fc.Target_poisoning]
    t1=time.time()
    lst=[[],[],[]]
    for rootsize in [10,30,100,300,1000]:
        for i in range(3):
            lst[i].append(exp(Cs, Cc, Cp, Ca, rootsize,attackL[i]))
            with open('pclog.txt','a',encoding='utf-8') as f:
                ct=time.time()-t1
                t1=time.time()
                f.write("{:.2f}".format(ct)+'\n')
    x=torch.tensor(lst)
    y=torch.tensor([[10,30,100,300,1000]])
    x=torch.cat((x,y))
    torch.save(x,'rootsize.pt')

    lst=[[],[],[]]
    for Cs in range(2,21,2):
        for i in range(3):
            lst[i].append(exp(Cs, Cc, Cp, Ca, rootsize, attackL[i]))
    x = torch.tensor(lst)
    y = torch.tensor([range(2,21,2)])
    x = torch.cat((x, y))
    torch.save(x, 'Cs.pt')

    lst=[[],[],[]]
    for Cc in range(0, 21, 2):
        for i in range(3):
            lst[i].append(exp(Cs, Cc, Cp, Ca, rootsize, attackL[i]))
    x = torch.tensor(lst)
    y = torch.tensor([range(0, 21, 2)])
    x = torch.cat((x, y))
    torch.save(x, 'Cc.pt')

    lst=[[],[],[]]
    for Cp in range(10, 100, 5):
        for i in range(3):
            lst[i].append(exp(Cs, Cc, Cp, Ca, rootsize, attackL[i]))
    x = torch.tensor(lst)
    y = torch.tensor([range(10, 100, 5)])/10
    x = torch.cat((x, y))
    torch.save(x, 'Cp.pt')

    lst=[[],[],[]]
    for Cp in range(10, 100, 5):
        for i in range(3):
            lst[i].append(exp(Cs, Cc, Cp, Ca, rootsize, attackL[i]))
    x = torch.tensor(lst)
    y = torch.tensor([range(10, 100, 5)]) / 10
    x = torch.cat((x, y))
    torch.save(x, 'Cpa.pt')





