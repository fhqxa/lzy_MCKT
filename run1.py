import numpy as np
import random
import torch
from torch import optim
from torchvision import transforms, datasets
from tqdm import tqdm
from datetime import datetime
import csv
import os
import shutil
from tqdm import trange
from sklearn.metrics import confusion_matrix

import cluster0
import criterion0
import dataset0
import label0
import net0, net1
from run0 import *
import utility0


if True:
    id_name_log = utility0.IDName()

    torch.manual_seed(SEED)  # cpu
    torch.cuda.manual_seed(SEED)  # gpu
    torch.cuda.manual_seed_all(SEED)  # multi-gpu
    np.random.seed(SEED)  # numpy
    random.seed(SEED)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # True: the speed of network is quicker than False, but failing to reproduce
    # def worker_init_fn(worker_id):  # num_workers
    #     np.random.seed(SEED + worker_id)

    time_start = datetime.now()

image_path, image_path_test, transform_train, transform_test, num_f_classes, num_c_classes, relation_f, \
eval_tree = dataset0.AllDatasets()
if image_path_test == 'None':  # split root to train and test
    labels = [i[1] for i in image_path]
    _, c = utility0.Unique1(labels, False)
    kind = 'balanced'  # 'balanced', 'longtailed.  perclass_test
    ratio_train_test = 6  # test = 1 / 6
    if kind == 'balanced':
        perclass_test = [min(c) // ratio_train_test] * len(c)
    elif kind == 'longtailed':
        perclass_test = [i // ratio_train_test for i in c]

    image_path_test, index_test = [], []
    perclass_test1 = perclass_test.copy()
    for i in range(len(image_path)):
        if perclass_test1[image_path[i][1]] > 0:
            image_path_test.append(image_path[i])
            index_test.append(i)
            perclass_test1[image_path[i][1]] -= 1
    index_test.reverse()
    for i in index_test:
        image_path.pop(i)
if 'LT' in switch_imbalanced:
    labels = [i[1] for i in image_path]
    _, perclass = utility0.Unique1(labels, False)
    perclass = [min(perclass)] * len(perclass)
    perclass = dataset0.LongTailDistribution(perclass, ratio_LT)
    selected, perclass1 = [], perclass.copy()
    for i in range(len(labels)):
        if perclass1[labels[i]] > 0:
            selected.append(i)
            perclass1[labels[i]] -= 1
    image_path = [image_path[i] for i in selected]
train_dataset = dataset0.DatasetFromPath(image_path, transform_train)
test_dataset = dataset0.DatasetFromPath(image_path_test, transform_test)

train_labels, test_labels = [i[1] for i in image_path], [i[1] for i in image_path_test]
_, num_perclass_train, num_perclass_test, _ = dataset0.NumPerclass(train_labels, test_labels)

# create dataset
path_create_dataset = f'{path_all_dataset}ZhaoWei/{switch_dataset}/'
path_create_dataset = False
if path_create_dataset:
    def f1_create_dataset(path, image_path):
        shutil.rmtree(path) if os.path.exists(path) else ''
        for i in trange(len(image_path)):
            fine = image_path[i][0].rsplit('/', 2)[1]
            path_fine = path + f'{fine}/'
            os.makedirs(path_fine) if not os.path.exists(path_fine) else ''
            image = image_path[i][0].rsplit('/', 1)[1]
            path_image = path_fine + image

            shutil.copyfile(image_path[i][0], path_image)

    f1_create_dataset(path_create_dataset + 'train/', image_path)
    f1_create_dataset(path_create_dataset + 'test/', image_path_test)
    e

# =====================================================================================================
head_f_marks2, tail_f_marks2 = label0.HeadTail(num_f_classes, alpha_hier_tail)

if beta_semantic_minor != 0 and alpha_hier_tail != 0:  # semantic
    relation_f_semantic = label0.TailFineToCoarse([i for i in range(num_f_classes)],
                                                  head_f_marks2, tail_f_marks2, relation_f)
    relation_f_semantic = label0.TailFineToCoarse_miss(relation_f_semantic)
    a, b = np.unique(np.array(relation_f_semantic), return_counts=True)
    num_c_classes = len(a)

    relation_f = relation_f_semantic
if beta_semantic_minor != 100 and alpha_hier_tail != 0:  # cluster
    relation_f_cluster = '''
19 18  4 11 14  8  7  7  3  2  8  4  6 10 12 11  2  6 12 11  2 15 15  9
7  8 17 14 15 17 16 11 19  0 18  4 19  6 11 13 15  3  0 15  7 17  4  5
3  9 14 12  5 19  1 14  5 19 10  5  9  8  1 14 15  0 15 16  9  9  1  9
16 16 14 12  9 12 13  7  0 10  1  1  8 17 13  2  0 10  6 18  1 16  2 16
5 15  4 13
     '''
    relation_f_cluster = cluster0.StrToList_relation(relation_f_cluster)
    a, _ = np.unique(np.array(relation_f_cluster), return_counts=True)
    num_c_classes = len(a)

    relation_f = relation_f_cluster

num_perclass_train_c = dataset0.NumPerclass_Coarse(num_f_classes, num_c_classes, relation_f, num_perclass_train)

# =====================================================================================================
if True:
    if switch_net == 'net0':
        model = net0.resnet32_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    elif switch_net == 'net1':
        model = net1.ResNet18_hier(num_f_classes, num_c_classes).to(device)
    elif switch_net == 'net2':
        model = net0.resnet20_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    elif switch_net == 'net3':
        model = net0.resnet44_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    elif switch_net == 'net4':
        model = net0.resnet56_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    elif switch_net == 'net5':
        model = net0.resnet110_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    elif switch_net == 'net6':
        model = net0.resnet1202_hier(num_f_classes, num_c_classes, LDAM_net).to(device)
    else:
        raise ValueError(switch_net)

    shutil.rmtree(directory_log) if os.path.exists(directory_log) else ''  # os.remove()  删除非空文件夹会拒绝访问
    os.makedirs(directory_log) if not os.path.exists(directory_log) else ''
    txt_name_log = f'log_{id_name_log}.txt'
    file_txt_log = directory_log + txt_name_log
    os.remove(file_txt_log) if os.path.exists(file_txt_log) else ''
    if not isTest:
        csv_name_log = f'log_{id_name_log}.csv'
        file_csv_log = directory_log + csv_name_log
        os.remove(file_csv_log) if os.path.exists(file_csv_log) else ''

        csv_header_log = ['L_', 'L', 'L_f_', 'L_f', 'L_c_', 'L_c', 'lr', 'A_c_e_', 'A_c_e', 'A_f_e_',
                          'A_f_e', 'A_f_e_top5', 'TIE_f_e', 'FH_f_e']
        with open(file_csv_log, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(csv_header_log)

        csv_name_cf_matrix = 'cf_matrix.csv'
        file_csv_cf_matrix = directory_log + csv_name_cf_matrix
        os.remove(file_csv_cf_matrix) if os.path.exists(file_csv_cf_matrix) else ''

if not isTest:
    if train_a == 'pre_hier' or train_a == 'pre_fine':
        utility0.LoadModel_pre_training(model, file_txt_log, load_dir, load_switch, load_str_epoch, load_int_epoch)
else:
    utility0.LoadModel_test(model, directory_test_model, test_epoch)

# =====================================================================================================
if True:
    width_txt_row = 120
    width_pb_left = 1
    width_msg_test = 8

    best_A_c_e = 0
    best_A_f_e = 0
    best_A_f_e_top5 = 0
    best_TIE_f_e = 1
    best_FH_f_e = 0
    best_A_hier = 0

    record_model = []  # best and final model

    feature_level = -1

    flag_c = 1
    flag_h = 1
    flag_f = 1


def Train(model, data_loader):
    s_l, s_l_f, s_l_c = 0, 0, 0
    s_a_f_e, s_a_c_e = 0, 0
    s_n_f_e = 0

    my_lambda = utility0.GetLambda(epoch)
    model.train()
    for data, target in data_loader:
        batch_size = target.size(0)
        s_n_f_e += batch_size

        data, label_f_e = data.to(device), target.long().to(device)  # torch.Size([128, 3, 32, 32])
        label_c_e = label0.FineToCoarse(label_f_e, relation_f)
        label_c_e = torch.from_numpy(label_c_e).long().to(device)

        out_f, out_c, _ = model(data)

        if my_lambda == 0:  # fine
            l_f = criterion_f(out_f, label_f_e)
            l_c = criterion_c(out_c, label_c_e).detach()
            l = l_f
        elif my_lambda == 1:  # coarse
            l_f = criterion_f(out_f, label_f_e).detach()
            l_c = criterion_c(out_c, label_c_e)
            l = l_c
        else:
            l_f = criterion_f(out_f, label_f_e)
            l_c = criterion_c(out_c, label_c_e)
            scale = utility0.GetScalingfac(l_c, l_f)
            l = my_lambda * l_c + (1 - my_lambda) * scale * l_f

        s_l += l.item() * batch_size
        s_l_f += l_f.item() * batch_size
        s_l_c += l_c.item() * batch_size

        topk_f = (1,)
        _, pred_f = out_f.topk(max(topk_f), 1, True, True)
        correct_f = pred_f.eq(label_f_e.view(-1, 1).expand_as(pred_f))
        acc_f_topk = []
        for k in topk_f:
            correct_f_k = correct_f[:, : k].reshape(-1).float().sum()
            acc_f_topk.append(correct_f_k)
        s_a_f_e += acc_f_topk[0].item()

        topk_c = (1,)
        _, pred_c = out_c.topk(max(topk_c), 1, True, True)
        correct_c = pred_c.eq(label_c_e.view(-1, 1).expand_as(pred_c))
        acc_c_topk = []
        for k in topk_c:
            correct_c_k = correct_c[:, : k].reshape(-1).float().sum()
            acc_c_topk.append(correct_c_k)
        s_a_c_e += acc_c_topk[0].item()

        # Back propagate
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    res = {'L_': s_l / s_n_f_e, 'L_c_': s_l_c / s_n_f_e, 'L_f_': s_l_f / s_n_f_e,
           'A_c_e_': s_a_c_e / s_n_f_e, 'A_f_e_': s_a_f_e / s_n_f_e}
    return res


def Test(model, data_loader):
    s_l, s_l_f, s_l_c = 0, 0, 0
    s_a_f_e, s_a_f_e_top5, s_a_c_e = 0, 0, 0
    s_n_f_e = 0

    my_lambda = utility0.GetLambda(epoch)
    all_label_f_e = []
    all_pred_f = []

    flag_feature = 0

    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            batch_size = target.size(0)
            s_n_f_e += batch_size

            data, label_f_e = data.to(device), target.long().to(device)
            label_c_e = label0.FineToCoarse(label_f_e, relation_f)
            label_c_e = torch.from_numpy(label_c_e).long().to(device)

            out_f, out_c, feature = model(data)

            if isCluster:
                feature_level = 5

                feature = feature[feature_level].cpu().numpy()
                labels_fea = label_f_e.cpu().numpy()
                if flag_feature == 0:
                    flag_feature = 1
                    utility0.SaveLog('feature_level: {}'.format(feature_level), file_txt_log) if isCluster else ''

                    feature_entire = feature
                    labels_fea_entire = labels_fea
                else:
                    feature_entire = np.concatenate(([feature_entire, feature]), axis=0)
                    labels_fea_entire = np.concatenate(([labels_fea_entire, labels_fea]), axis=0)

            if my_lambda == 0:  # fine
                l_f = criterion_f(out_f, label_f_e)
                l_c = criterion_c(out_c, label_c_e).detach()
                l = l_f
            elif my_lambda == 1:  # coarse
                l_f = criterion_f(out_f, label_f_e).detach()
                l_c = criterion_c(out_c, label_c_e)
                l = l_c
            else:
                l_f = criterion_f(out_f, label_f_e)
                l_c = criterion_c(out_c, label_c_e)
                scale = utility0.GetScalingfac(l_c, l_f)
                l = my_lambda * l_c + (1 - my_lambda) * scale * l_f

            s_l += l.item() * batch_size
            s_l_f += l_f.item() * batch_size
            s_l_c += l_c.item() * batch_size

            topk_f = (1, 5)
            _, pred_f = out_f.topk(max(topk_f), 1, True, True)  # topk(): k; dim; largest; sorted
            correct_f = pred_f.eq(label_f_e.view(-1, 1).expand_as(pred_f))
            acc_f_topk = []
            for k in topk_f:
                correct_f_k = correct_f[:, : k].reshape(-1).float().sum()
                acc_f_topk.append(correct_f_k)
            s_a_f_e += acc_f_topk[0].item()
            s_a_f_e_top5 += acc_f_topk[1].item()

            topk_c = (1,)
            _, pred_c = out_c.topk(max(topk_c), 1, True, True)
            correct_c = pred_c.eq(label_c_e.view(-1, 1).expand_as(pred_c))
            acc_c_topk = []
            for k in topk_c:
                correct_c_k = correct_c[:, : k].reshape(-1).float().sum()
                acc_c_topk.append(correct_c_k)
            s_a_c_e += acc_c_topk[0].item()

            all_label_f_e.extend(label_f_e.cpu().numpy().flatten())
            all_pred_f.extend(pred_f[:, : 1].cpu().numpy().flatten())

        # np.set_printoptions(threshold=np.inf)  # no default print
        cf_matrix = confusion_matrix(all_label_f_e, all_pred_f)
        cls_hit_tie = utility0.EvaHier_TreeInducedError(eval_tree, all_pred_f, all_label_f_e)
        cls_hit_fh = utility0.EvaHier_HierarchicalPrecisionAndRecall(eval_tree, all_pred_f, all_label_f_e)

        res = {'L': s_l / s_n_f_e, 'L_c': s_l_c / s_n_f_e, 'L_f': s_l_f / s_n_f_e, 'A_c_e': s_a_c_e / s_n_f_e,
               'A_f_e': s_a_f_e / s_n_f_e, 'A_f_e_top5': s_a_f_e_top5 / s_n_f_e,
               'TIE_f_e':sum(cls_hit_tie) / s_n_f_e, 'FH_f_e':sum(cls_hit_fh) / s_n_f_e}

        msg = 'Test...\n' \
              'L: {:.4f}, L_c: {:.4f}, L_f: {:.4f}, A_c_e: {:.4f}, ' \
              '\nA_f_e: {:.4f}, A_f_e_top5: {:.4f}, TIE_f_e: {:.4f}, FH_f_e: {:.4f}'.format(
            res['L'], res['L_f'], res['L_c'], res['A_c_e'],
            res['A_f_e'], res['A_f_e_top5'], res['TIE_f_e'], res['FH_f_e'])

    return res, (cf_matrix, cls_hit_tie, cls_hit_fh), msg if not isCluster else (feature_entire, labels_fea_entire)


if train_a == 'coarse':
    progress_bar = tqdm(range(coarse_epoch))
    for epoch in progress_bar:
        if flag_c == 1:
            flag_c = 0

            batch_size = batch_size1
            beta_EffectNum_f = beta_EffectNum_f1
            beta_EffectNum_c = beta_EffectNum_c1
            gamma_Focal = gamma_Focal1

            defer_epoch = defer_epoch1
            switch_defer = switch_defer0
            switch_resampling = switch_resampling0
            switch_reweighting = switch_reweighting0
            switch_criterion = switch_criterion0
            switch_optimizer = switch_optimizer0

            train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                defer_epoch, batch_size, epoch)

            criterion_f, criterion_c = criterion0.Criterion(
                num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

            if switch_optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
                # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
            elif switch_optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=1e-5)
            elif switch_optimizer == 'Adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
            else:
                raise ValueError(switch_optimizer)

        lr = utility0.AdjustLearningRate(epoch, optimizer)

        if switch_defer == 'DeferRS' and epoch == defer_epoch:
            train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling, defer_epoch,
                batch_size, epoch)
        if switch_defer == 'DeferRW' and epoch == defer_epoch:
            criterion_f, criterion_c = criterion0.Criterion(
                num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

        res_train = Train(model, train_loader) if not isTest else ''
        res_test, _, msg_test = Test(model, test_loader)

        if not isTest:
            res_test.update(res_train)
            res_test.update({'lr': lr})
            utility0.SaveCsv(res_test, file_csv_log, csv_header_log)

            if res_test['A_c_e'] > best_A_c_e:
                best_ep_A_c_e = epoch
                best_A_c_e = res_test['A_c_e']
                utility0.SaveModel(model, best_ep_A_c_e, -1)

            if epoch == coarse_epoch - 1:
                record_model.append(best_ep_A_c_e)
                if epoch != best_ep_A_c_e:
                    utility0.SaveModel(model, epoch, record_model)
                    record_model.append(epoch)
        else:
            record_test.append(res_test['A_c_e'])
            test_number -= 1
            if test_number <= 0:
                break

        progress_bar.set_description(msg_test.replace('\n', ''))
        progress_bar.set_postfix(CUDA=CUDA, Dir=directory_log.split(f'{os.getcwd()}_log/log/')[1],
                                 Epoch=f'{epoch}/{end_epoch - 1}')  # auto-sorted by ASCII code

    utility0.DrawOnCsv_loss('L_c', file_csv_log, id_name_log)

elif train_a == 'hier' or train_a == 'pre_hier' or train_a == 'coarse_hier':
    progress_bar = tqdm(range(end_epoch))
    for epoch in progress_bar:
        if train_a == 'pre_hier':
            epoch += coarse_epoch

        if epoch < coarse_epoch:  # c
            if flag_c == 1:
                flag_c = 0

                batch_size = batch_size1
                beta_EffectNum_f = beta_EffectNum_f1
                beta_EffectNum_c = beta_EffectNum_c1
                gamma_Focal = gamma_Focal1

                defer_epoch = defer_epoch1
                switch_defer = switch_defer0
                switch_resampling = switch_resampling0
                switch_reweighting = switch_reweighting0
                switch_criterion = switch_criterion0
                switch_optimizer = switch_optimizer0

                train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                    train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                    defer_epoch, batch_size, epoch)

                criterion_f, criterion_c = criterion0.Criterion(
                    num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                    defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

                if switch_optimizer == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
                elif switch_optimizer == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=1e-5)
                elif switch_optimizer == 'Adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
                else:
                    raise ValueError(switch_optimizer)
        elif epoch >= coarse_epoch + hier_epoch:  # f
            if flag_f == 1:
                flag_f = 0

                batch_size = batch_size3
                beta_EffectNum_f = beta_EffectNum_f3
                beta_EffectNum_c = beta_EffectNum_c3
                gamma_Focal = gamma_Focal3

                defer_epoch = defer_epoch3
                switch_defer = switch_defer2
                switch_resampling = switch_resampling2
                switch_reweighting = switch_reweighting2
                switch_criterion = switch_criterion2
                switch_optimizer = switch_optimizer2

                train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                    train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                    defer_epoch, batch_size, epoch)

                criterion_f, criterion_c = criterion0.Criterion(
                    num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                    defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

                if switch_optimizer == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
                elif switch_optimizer == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=1e-5)
                elif switch_optimizer == 'Adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
                else:
                    raise ValueError(switch_optimizer)
        else:
            if flag_h == 1:
                flag_h = 0

                batch_size = batch_size2
                beta_EffectNum_f = beta_EffectNum_f2
                beta_EffectNum_c = beta_EffectNum_c2
                gamma_Focal = gamma_Focal2

                defer_epoch = defer_epoch2
                switch_defer = switch_defer1
                switch_resampling = switch_resampling1
                switch_reweighting = switch_reweighting1
                switch_criterion = switch_criterion1
                switch_optimizer = switch_optimizer1

                train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                    train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                    defer_epoch, batch_size, epoch)

                criterion_f, criterion_c = criterion0.Criterion(
                    num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                    defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

                if switch_optimizer == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
                elif switch_optimizer == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=1e-5)
                elif switch_optimizer == 'Adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
                else:
                    raise ValueError(switch_optimizer)

        lr = utility0.AdjustLearningRate(epoch, optimizer)

        if switch_defer == 'DeferRS' and epoch == defer_epoch:
            train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                defer_epoch, batch_size, epoch)
        if switch_defer == 'DeferRW' and epoch == defer_epoch:
            criterion_f, criterion_c = criterion0.Criterion(
                num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

        res_train = Train(model, train_loader) if not isTest else ''
        res_test, cf_matrix, msg_test = Test(model, test_loader)

        if not isTest:
            res_test.update(res_train)
            res_test.update({'lr': lr})
            utility0.SaveCsv(res_test, file_csv_log, csv_header_log)

            if epoch < coarse_epoch:  # coarse
                if res_test['A_c_e'] > best_A_c_e:
                    best_ep_A_c_e = epoch
                    best_A_c_e = res_test['A_c_e']
                    utility0.SaveModel(model, best_ep_A_c_e, -1)
            elif epoch >= coarse_epoch + hier_epoch:  # fine
                if res_test['A_f_e'] > best_A_f_e:
                    best_ep_A_f_e = epoch
                    best_A_f_e = res_test['A_f_e']
                    utility0.SaveModel(model, best_ep_A_f_e, record_model)

                    best_msg_test = msg_test
                    best_cf_matrix = cf_matrix
                if res_test['A_f_e_top5'] > best_A_f_e_top5:
                    best_ep_A_f_e_top5 = epoch
                    best_A_f_e_top5 = res_test['A_f_e_top5']

                if res_test['TIE_f_e'] < best_TIE_f_e:
                    best_ep_TIE_f_e = epoch
                    best_TIE_f_e = res_test['TIE_f_e']

                if res_test['FH_f_e'] > best_FH_f_e:
                    best_ep_FH_f_e = epoch
                    best_FH_f_e = res_test['FH_f_e']
            else:  # hier
                if res_test['A_f_e'] > best_A_hier:
                    best_ep_A_hier = epoch
                    best_A_hier = res_test['A_f_e']
                    utility0.SaveModel(model, best_ep_A_hier, record_model)

            if epoch == coarse_epoch - 1:
                record_model.append(best_ep_A_c_e)
                if epoch != best_ep_A_c_e:
                    utility0.SaveModel(model, epoch, record_model)
                    record_model.append(epoch)
            elif epoch == coarse_epoch + hier_epoch - 1:
                record_model.append(best_ep_A_hier)
                if epoch != best_ep_A_hier:
                    utility0.SaveModel(model, epoch, record_model)
                    record_model.append(epoch)

            if train_a == 'pre_hier' or train_a == 'coarse_hier':
                if epoch == coarse_epoch + hier_epoch - 1:
                    break
        else:
            record_test.append(res_test['A_f_e'])
            test_number -= 1
            if test_number <= 0:
                break

        progress_bar.set_description(msg_test.replace('\n', ''))
        progress_bar.set_postfix(CUDA=CUDA, Epoch=f'{epoch}/{end_epoch - 1}')

    utility0.DrawOnCsv_loss('L', file_csv_log, id_name_log)

elif train_a == 'fine' or train_a == 'pre_fine':
    progress_bar = tqdm(range(coarse_epoch + hier_epoch, end_epoch))
    for epoch in progress_bar:
        feature_level += 1

        if flag_f == 1:
            flag_f = 0

            batch_size = batch_size3
            beta_EffectNum_f = beta_EffectNum_f3
            beta_EffectNum_c = beta_EffectNum_c3
            gamma_Focal = gamma_Focal3

            defer_epoch = defer_epoch3
            switch_defer = switch_defer2
            switch_resampling = switch_resampling2
            switch_reweighting = switch_reweighting2
            switch_criterion = switch_criterion2
            switch_optimizer = switch_optimizer2

            train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                defer_epoch, batch_size, epoch)

            criterion_f, criterion_c = criterion0.Criterion(
                num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

            if switch_optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
                # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
            elif switch_optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=1e-5)
            elif switch_optimizer == 'Adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
            else:
                raise ValueError(switch_optimizer)

        lr = utility0.AdjustLearningRate(epoch, optimizer)

        if switch_defer == 'DeferRS' and epoch == defer_epoch:
            train_loader, test_loader, val_loader = dataset0.DataloaderFromDataset(
                train_dataset, test_dataset, num_perclass_train, num_perclass_test, switch_resampling,
                defer_epoch, batch_size, epoch)
        if switch_defer == 'DeferRW' and epoch == defer_epoch:
            criterion_f, criterion_c = criterion0.Criterion(
                num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting,
                defer_epoch, beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch)

        res_train = Train(model, train_loader) if not isTest else ''
        res_test, cf_matrix, msg_test = Test(model, test_loader) if not isCluster else Test(model, test_loader)

        if isCluster:
            feature, labels = msg_test
            # print(feature.shape, labels.shape) # (10847, 64, 8, 8) (10847,)  64*8*8=4096

            # feature = feature.reshape(-1, 4096)
            # data = np.column_stack((feature, labels))
            # # print(data.shape)
            # import scipy.io as io
            # mat_path = 'cifar100_original_4096_test'
            # io.savemat(mat_path, {'data': data})
            # e

            feature = feature.reshape(feature.shape[0], -1)
            relation_f = cluster0.ClusterCoarse(feature, labels, num_f_classes, num_c_classes)
            if True:
                utility0.PrintOnRowWidth('relation_f: ' + str(relation_f), width_txt_row, file_txt_log)
                utility0.SaveLog('fine_to_coarse: c_label(c_num) [f_label][[f_num]]', file_txt_log, False)
                name_f_classes = \
                    ['苹果', '水族馆鱼', '宝贝', '熊', '海狸', '床', '蜜蜂', '甲虫', '自行车', '瓶子', '碗', '男孩', '桥',
                     '公共汽车', '蝴蝶', '骆驼', '罐', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟', '云', '蟑螂', '沙发',
                     '螃蟹', '鳄鱼', '杯子', '恐龙', '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠', '房子', '袋鼠',
                     '键盘', '台灯', '割草机', '豹', '狮子', '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠', '蘑菇',
                     '橡树', '橘子', '兰花', '水獭', '棕榈树', '梨', '皮卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
                     '负鼠', '兔子', '浣熊', '射线', '路', '火箭', '玫瑰', '海', '海豹', '鲨鱼', '地鼠', '臭鼬',
                     '摩天大楼', '蜗牛', '蛇', '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜辣椒', '桌子', '坦克', '电话', '电视机',
                     '老虎', '拖拉机', '火车', '鳟鱼', '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫']
                name_c_classes = \
                    ['水生哺乳动物', '鱼', '花卉', '食品容器', '水果和蔬菜', '家用电器', '家庭家具', '昆虫', '大型食肉动物',
                     '大型人造户外用品', '大自然户外场景', '大型杂食动物和食草动物', '中型哺乳动物', '非昆虫无脊椎动物',
                     '人', '爬行动物', '小型哺乳动物', '树木', '车辆一', '车辆二']
                utility0.VisualRelation_fine_to_coarse(name_c_classes, num_perclass_train_c, name_f_classes,
                                                       num_perclass_train, relation_f, file_txt_log, True)

        if not isTest:
            res_test.update(res_train)
            res_test.update({'lr': lr})
            utility0.SaveCsv(res_test, file_csv_log, csv_header_log)

            if res_test['A_f_e'] > best_A_f_e:
                best_ep_A_f_e = epoch
                best_A_f_e = res_test['A_f_e']
                utility0.SaveModel(model, best_ep_A_f_e, -1)

                best_msg_test = msg_test
                best_cf_matrix = cf_matrix
            if res_test['A_f_e_top5'] > best_A_f_e_top5:
                best_ep_A_f_e_top5 = epoch
                best_A_f_e_top5 = res_test['A_f_e_top5']

            if res_test['TIE_f_e'] < best_TIE_f_e:
                best_ep_TIE_f_e = epoch
                best_TIE_f_e = res_test['TIE_f_e']

            if res_test['FH_f_e'] > best_FH_f_e:
                best_ep_FH_f_e = epoch
                best_FH_f_e = res_test['FH_f_e']
        else:
            record_test.append(res_test['A_f_e'])
            test_number -= 1
            if test_number <= 0:
                break

        progress_bar.set_description(msg_test.replace('\n', ''))
        progress_bar.set_postfix(CUDA=CUDA, Epoch=f'{epoch}/{end_epoch - 1}')

    utility0.DrawOnCsv_loss('L_f', file_csv_log, id_name_log)

# =====================================================================================================
if not isTest:
    if True:
        time_end = datetime.now()

        if train_a == 'coarse':  # coarse
            utility0.SaveLog('Save... epoch - acc: c: {} - {:.4f}'.format(best_ep_A_c_e, best_A_c_e), file_txt_log)
        elif train_a == 'hier':  # hier
            utility0.SaveLog('Save... epoch - acc: c: {} - {:.4f}; h: {} - {:.4f}; '
                             'f: {} - {:.4f}, epoch - acctop5: {} - {:.4f},\n'
                             'epoch - tie: {} - {:.4f}, epoch - fh: {} - {:.4f}'.format(
                best_ep_A_c_e, best_A_c_e, best_ep_A_hier, best_A_hier,
                best_ep_A_f_e, best_A_f_e, best_ep_A_f_e_top5, best_A_f_e_top5,
                best_ep_TIE_f_e, best_TIE_f_e, best_ep_FH_f_e, best_FH_f_e
            ), file_txt_log)
            utility0.SaveLog('{}'.format(best_msg_test), file_txt_log, False)

            cf_matrix, cls_hit_tie, cls_hit_fh = best_cf_matrix[0], best_cf_matrix[1], best_cf_matrix[2]
            cls_hit_acc, cls_cnt = np.diag(cf_matrix), cf_matrix.sum(axis=1)
            acc_head, acc_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_acc, cls_cnt)
            tie_head, tie_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_tie, cls_cnt)
            fh_head, fh_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_fh, cls_cnt)
            utility0.SaveLog('20% head / tail\nacc: {:.4f} / {:.4f}, tie: {:.4f} / {:.4f}, fh: {:.4f} / {:.4f}'.format(
                acc_head, acc_tail, tie_head, tie_tail, fh_head, fh_tail), file_txt_log, False)
            utility0.SaveLog('cls_hit_acc: \n{}\ncls_hit_tie: \n{}\ncls_hit_fh: \n{}\ncls_cnt: \n{}'.format(
                cls_hit_acc, cls_hit_tie, cls_hit_fh, cls_cnt), file_txt_log, False)
            np.savetxt(file_csv_cf_matrix, cf_matrix, delimiter=',')
        elif train_a == 'fine' or train_a == 'pre_fine':  # fine; pre_fine
            utility0.SaveLog('Save... epoch - acc: f: {} - {:.4f}, epoch - acctop5: {} - {:.4f},\n'
                             'epoch - tie: {} - {:.4f}, epoch - fh: {} - {:.4f}'.format(
                best_ep_A_f_e, best_A_f_e, best_ep_A_f_e_top5, best_A_f_e_top5,
                best_ep_TIE_f_e, best_TIE_f_e, best_ep_FH_f_e, best_FH_f_e), file_txt_log)
            utility0.SaveLog('{}'.format(best_msg_test), file_txt_log, False)

            cf_matrix, cls_hit_tie, cls_hit_fh = best_cf_matrix[0], best_cf_matrix[1], best_cf_matrix[2]
            cls_hit_acc, cls_cnt = np.diag(cf_matrix), cf_matrix.sum(axis=1)
            acc_head, acc_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_acc, cls_cnt)
            tie_head, tie_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_tie, cls_cnt)
            fh_head, fh_tail = utility0.EvalHeadTail(0.2, 0.2, cls_hit_fh, cls_cnt)
            utility0.SaveLog('20% head / tail\nacc: {:.4f} / {:.4f}, tie: {:.4f} / {:.4f}, fh: {:.4f} / {:.4f}'.format(
                acc_head, acc_tail, tie_head, tie_tail, fh_head, fh_tail), file_txt_log, False)
            utility0.SaveLog('cls_hit_acc: \n{}\ncls_hit_tie: \n{}\ncls_hit_fh: \n{}\ncls_cnt: \n{}'.format(
                cls_hit_acc, cls_hit_tie, cls_hit_fh, cls_cnt), file_txt_log, False)
            np.savetxt(file_csv_cf_matrix, cf_matrix, delimiter=',')
        elif train_a == 'pre_hier':  # pre_hier
            utility0.SaveLog('Save... epoch - acc: h: {} - {:.4f}'.format(best_ep_A_hier, best_A_hier), file_txt_log)
        elif train_a == 'coarse_hier':  # coarse_hier
            utility0.SaveLog('Save... epoch - acc: c: {} - {:.4f}; h: {} - {:.4f}'.format(
                best_ep_A_c_e, best_A_c_e, best_ep_A_hier, best_A_hier), file_txt_log)

        utility0.SaveLog('{}'.format('=' * 120), file_txt_log, False)

        utility0.PrintOnRowWidth('num_perclass_train: ' + str(num_perclass_train), width_txt_row, file_txt_log)
        utility0.PrintOnRowWidth('num_perclass_test: ' + str(num_perclass_test), width_txt_row, file_txt_log)
        utility0.PrintOnRowWidth('num_perclass_train_c: ' + str(num_perclass_train_c), width_txt_row, file_txt_log)
        utility0.PrintOnRowWidth('relation_f: ' + str(relation_f), width_txt_row, file_txt_log)

        utility0.SaveLog('fine_to_coarse: c_label(c_num) [f_label][[f_num]]', file_txt_log, False)

        name_c_classes = [i for i in range(num_c_classes)]
        name_f_classes = [i for i in range(num_f_classes)]
        utility0.VisualRelation_fine_to_coarse(name_c_classes, num_perclass_train_c, name_f_classes,
                                               num_perclass_train, relation_f, file_txt_log, False)
        if alpha_hier_tail == 100 and beta_semantic_minor == 100:
            if switch_dataset == 'Cifar-100_tree':
                name_c_classes = \
                    ['水生哺乳动物', '鱼', '花卉', '食品容器', '水果和蔬菜', '家用电器', '家庭家具', '昆虫', '大型食肉动物',
                     '大型人造户外用品', '大自然户外场景', '大型杂食动物和食草动物', '中型哺乳动物', '非昆虫无脊椎动物',
                     '人', '爬行动物', '小型哺乳动物', '树木', '车辆一', '车辆二']
                name_f_classes = \
                    ['苹果', '水族馆鱼', '宝贝', '熊', '海狸', '床', '蜜蜂', '甲虫', '自行车', '瓶子', '碗', '男孩', '桥',
                     '公共汽车', '蝴蝶', '骆驼', '罐', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟', '云', '蟑螂', '沙发',
                     '螃蟹', '鳄鱼', '杯子', '恐龙', '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠', '房子', '袋鼠',
                     '键盘', '台灯', '割草机', '豹', '狮子', '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠', '蘑菇',
                     '橡树', '橘子', '兰花', '水獭', '棕榈树', '梨', '皮卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
                     '负鼠', '兔子', '浣熊', '鳐', '路', '火箭', '玫瑰', '海', '海豹', '鲨鱼', '地鼠', '臭鼬',
                     '摩天大楼', '蜗牛', '蛇', '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜辣椒', '桌子', '坦克', '电话', '电视机',
                     '老虎', '拖拉机', '火车', '鳟鱼', '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫']
                utility0.VisualRelation_fine_to_coarse(name_c_classes, num_perclass_train_c, name_f_classes,
                                                       num_perclass_train, relation_f, file_txt_log, True)

        utility0.VisualLearningRate(file_txt_log, file_csv_log)

        utility0.SaveLog(
            '{}\nEpochTime: {}, Start: {} - End: {},\n'
            'train_a: {}, switch_imbalanced: {}, switch_augument: {}, switch_net: {}, \n'
            'coarse_epoch: {}, hier_epoch: {}, fine_epoch: {}, end_epoch: {}, epoch: 0~{}~{}~{}, '
            'defer_epoch: c{}h{}f{},\n'
            'batch_size: c{}h{}f{}, batch_size_test: {}(shuffle: {}),\n'
            'beta_EffectNum_f: c{}h{}f{}, beta_EffectNum_c: c{}h{}f{}, gamma_Focal: c{}h{}f{}'.format(
                '=' * 120, time_end - time_start, time_start, time_end,
                train_a, switch_imbalanced, switch_augument, switch_net,
                coarse_epoch, hier_epoch, fine_epoch, end_epoch, coarse_epoch, coarse_epoch + hier_epoch, end_epoch,
                defer_epoch1, defer_epoch2, defer_epoch3,
                batch_size1, batch_size2, batch_size3, batch_size_test, shuffle_test,
                beta_EffectNum_f1, beta_EffectNum_f2, beta_EffectNum_f3,
                beta_EffectNum_c1, beta_EffectNum_c2, beta_EffectNum_c3,
                gamma_Focal1, gamma_Focal2, gamma_Focal3), file_txt_log, False)

    # -------------------------------------------------------------------------------------------------------------
    utility0.SaveLog('{}\n{}\ndirectory_log: {}, \nid_name_log: {}, seed: {}, \nimage_path: {}'.format(
        '=' * 120, name_switch, directory_log, id_name_log, SEED, image_path[0]), file_txt_log)

    if train_a == 'coarse':  # coarse
        str_acc = utility0.AccToString(best_A_c_e)
        os.rename(directory_log, directory_log.rstrip('/') + f'_c{str_acc}/')
    elif train_a == 'hier':  # hier
        str_acc1 = utility0.AccToString(best_A_c_e)
        str_acc2 = utility0.AccToString(best_A_hier)
        str_acc3 = utility0.AccToString(best_A_f_e)
        os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc3}/')
        # os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc3}_c{coarse_epoch}h{hier_epoch}f{fine_epoch}/')
    elif train_a == 'fine' or train_a == 'pre_fine':  # fine; pre_fine
        str_acc = utility0.AccToString(best_A_f_e)
        os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc}/')
    # elif train_a == 'pre_hier':  # pre_hier
    #     str_acc = utility0.AccToString(best_A_hier)
    #     os.rename(directory_log, directory_log.rstrip('/') + f'_h{str_acc}/')
    # elif train_a == 'coarse_hier':  # coarse_hier
    #     str_acc1 = utility0.AccToString(best_A_c_e)
    #     str_acc2 = utility0.AccToString(best_A_hier)
    #     os.rename(directory_log, directory_log.rstrip('/') + f'_f{str_acc1}h{str_acc2}/')
else:
    utility0.SaveLog(record_test, file_txt_log) if not isCluster else ''
