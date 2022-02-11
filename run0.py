                         # 'SUN397'-8,        'iNaturalist2017',
                         # 'VOC2007_Per'-4,   'VOC2012_Per',     'VOC2007_PerBir',  'VOC2012_PerBir'
switch_dataset      = 1  # 'CIFAR-10',        'CIFAR-100',       'tieredImageNet',  'ImageNet2012',
switch_net          = 0  # 'net0', 'net1', ...
alpha_hier_tail     = 0  # 0-0,20-1,40-2,60-3,80-4,100-5
beta_semantic_minor = 5 if alpha_hier_tail != 0 else 'x'  # 0-0,20-1,40-2,60-3,80-4,100-5
switch_my_lambda    = 5 if alpha_hier_tail != 0 else 'x'  # 'my_lambda0', 'my_lambda1', ...
# ======================== 0                  1                  2                  3
switch_criterion0   = 0  # 'CrossEntropy',    'Focal',           'LDAM',
switch_strategy0    = 0  # 'None'-0,'RS'-1,'SMOTE'-2,'DRS'-3,'DSMOTE'-4,'RW'-5,'CB'-6,'DRW'-7,'DCB'-8
switch_optimizer0   = 0  # 'SGD',             'Adam',            'Adagrad',
switch_lr0          = 5  # 'lr0', 'lr1', ...
# ---------------------------------------------------------------------------------------------
switch_criterion1   = 0  # 'CrossEntropy',    'Focal',           'LDAM',
switch_strategy1    = 0  # 'None'-0,'RS'-1,'SMOTE'-2,'DRS'-3,'DSMOTE'-4,'RW'-5,'CB'-6,'DRW'-7,'DCB'-8
switch_optimizer1   = 0  # 'SGD',             'Adam',            'Adagrad',
switch_lr1          = 5  # 'lr0', 'lr1', ...
# ---------------------------------------------------------------------------------------------
switch_criterion2   = 0  # 'CrossEntropy',    'Focal',           'LDAM',
switch_strategy2    = 0  # 'None'-0,'RS'-1,'SMOTE'-2,'DRS'-3,'DSMOTE'-4,'RW'-5,'CB'-6,'DRW'-7,'DCB'-8
switch_optimizer2   = 0  # 'SGD',             'Adam',            'Adagrad',
switch_lr2          = 0  # 'lr0', 'lr1', ...
CUDA, SEED = 1, 0  # 0;1, 0;None
coarse_epoch, hier_epoch, fine_epoch = 55, 45, 200
# coarse_epoch, hier_epoch, fine_epoch = 60, 60, 120
train_a = 1  # 'coarse'-0,'hier'-1,'fine'-2,'pre_hier'-3,'pre_fine'-4,'coarse_hier'-5,'flat'-6
if True:
    train_a, fine_epoch = (2, 300) if alpha_hier_tail == 0 else (train_a, fine_epoch)
    # train_a, fine_epoch = (2, 240) if alpha_hier_tail == 0 else (train_a, fine_epoch)
    switch_imbalanced = 3 if switch_dataset == 0 or switch_dataset == 1 or switch_dataset == 2 else 0
    switch_imbalanced = ['Original', 'LT_200', 'LT_100', 'LT_50', 'LT_20', 'LT_10'][switch_imbalanced]
    ratio_LT = int(switch_imbalanced.rsplit('_', 1)[1]) if 'LT' in switch_imbalanced else 'None'
    # switch_lr1 = 4 if switch_my_lambda == 1 else switch_lr1
    switch_augument = ['Augument0', 'Augument1', 'Augument2', 'Augument3'][0]
    switch_criterion0, switch_criterion1 = switch_criterion2, switch_criterion2
    switch_optimizer0, switch_optimizer1 = switch_optimizer2, switch_optimizer2

    switch = str(switch_dataset) + str(switch_net) + \
             str(alpha_hier_tail) + str(beta_semantic_minor) + str(switch_my_lambda) + \
             'c' + str(switch_criterion0) + str(switch_strategy0) + str(switch_optimizer0) + str(switch_lr0) + \
             'h' + str(switch_criterion1) + str(switch_strategy1) + str(switch_optimizer1) + str(switch_lr1) + \
             'f' + str(switch_criterion2) + str(switch_strategy2) + str(switch_optimizer2) + str(switch_lr2)

    switch_dataset      = ['CIFAR-10',        'CIFAR-100',       'tieredImageNet',  'ImageNet2012',
                           'VOC2007_Per',     'VOC2012_Per',     'VOC2007_PerBir',  'VOC2012_PerBir',
                           'SUN397',          'iNaturalist2017',
                                                                                                     ][int(switch[0])]
    switch_net          = ['net0', 'net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', 'net8',  ][int(switch[1])]
    alpha_hier_tail     = [0,                 20,                 40,               60,
                           80,                100                                                    ][int(switch[2])]
    # the percent of hier tail. For example, 'alpha=0' means all be flat; 'alpha=50' means one half tail be hier but
    # another half head be flat; 'alpha=100' means all be hier;
    beta_semantic_minor = [0,                 20,                 40,               60,
                           80,                100][int(switch[3])] if alpha_hier_tail != 0 else 'x'
    # the percent of semantic under hier tail. For example, 'alpha=100, beta=0' means all be hier, therein all major
    # be cluster; 'alpha=100, beta=50' means all be hier, therein one half minor be semantic but another half major
    # be cluster; 'alpha=80, beta=80' means 80(all) percent be hier, 64(all)/80(therein) percent minor be semantic
    # but 16(all)/20(therein) percent major be cluster, and 20(all) percent be flat; 'alpha=100, beta=100' means all
    # be hier, therein all minor be semantic;
    switch_my_lambda    = ['my_lambda0', 'my_lambda1', 'my_lambda2', 'my_lambda3', 'my_lambda4', 'my_lambda5',
                           'my_lambda6', 'my_lambda7', 'my_lambda8', 'my_lambda9',
                           '',][int(switch[4])] if alpha_hier_tail != 0 else 'x'
    # ==========================================================================================
    switch_criterion0   = ['CrossEntropy',    'Focal',           'LDAM',                            ][int(switch[6])]
    switch_strategy0    = ['None',            'RS',              'SMOTE',           'DRS',
                           'DSMOTE',          'RW',              'CB',              'DRW',
                           'DCB'                                                                    ][int(switch[7])]
    switch_optimizer0   = ['SGD',             'Adam',            'Adagrad',                         ][int(switch[8])]
    switch_lr0          = ['lr0', 'lr1', 'lr2', 'lr3', 'lr4', 'lr5', 'lr6', 'lr7', 'lr8', 'lr9'     ][int(switch[9])]
    # -------------------------------------------------------------------------------------------------------------
    switch_criterion1   = ['CrossEntropy',    'Focal',           'LDAM',                            ][int(switch[11])]
    switch_strategy1    = ['None',            'RS',              'SMOTE',           'DRS',
                           'DSMOTE',          'RW',              'CB',              'DRW',
                           'DCB'                                                                    ][int(switch[12])]
    switch_optimizer1   = ['SGD',             'Adam',            'Adagrad',                         ][int(switch[13])]
    switch_lr1          = ['lr0', 'lr1', 'lr2', 'lr3', 'lr4', 'lr5', 'lr6', 'lr7', 'lr8', 'lr9'     ][int(switch[14])]
    # -------------------------------------------------------------------------------------------------------------
    switch_criterion2   = ['CrossEntropy',    'Focal',           'LDAM',                            ][int(switch[16])]
    switch_strategy2    = ['None',            'RS',              'SMOTE',           'DRS',
                           'DSMOTE',          'RW',              'CB',              'DRW',
                           'DCB'                                                                    ][int(switch[17])]
    switch_optimizer2   = ['SGD',             'Adam',            'Adagrad',                         ][int(switch[18])]
    switch_lr2          = ['lr0', 'lr1', 'lr2', 'lr3', 'lr4', 'lr5', 'lr6', 'lr7', 'lr8', 'lr9'     ][int(switch[19])]

    name_switch = f'{switch_dataset}_{switch_net}_{switch_dataset}_' \
                  f'{alpha_hier_tail}_{beta_semantic_minor}_{switch_my_lambda}_' \
                  f'c_{switch_criterion0}_{switch_strategy0}_{switch_optimizer0}_{switch_lr0}_' \
                  f'h_{switch_criterion1}_{switch_strategy1}_{switch_optimizer1}_{switch_lr1}_' \
                  f'f_{switch_criterion2}_{switch_strategy2}_{switch_optimizer2}_{switch_lr2}'
    
    for i in range(3):
        if i == 0:
            switch_strategy = switch_strategy0
        elif i == 1:
            switch_strategy = switch_strategy1
        elif i == 2:
            switch_strategy = switch_strategy2

        if switch_strategy == 'None':
            switch_defer = 'NoDefer'
            switch_resampling = 'NoRS'
            switch_reweighting = 'NoRW'
        elif switch_strategy == 'RS':
            switch_defer = 'NoDefer'
            switch_resampling = 'Resampling'
            switch_reweighting = 'NoRW'
        elif switch_strategy == 'SMOTE':
            switch_defer = 'NoDefer'
            switch_resampling = 'SMOTE'
            switch_reweighting = 'NoRW'
        elif switch_strategy == 'DRS':
            switch_defer = 'DeferRS'
            switch_resampling = 'Resampling'
            switch_reweighting = 'NoRW'
        elif switch_strategy == 'DSMOTE':
            switch_defer = 'DeferRS'
            switch_resampling = 'SMOTE'
            switch_reweighting = 'NoRW'
        elif switch_strategy == 'RW':
            switch_defer = 'NoDefer'
            switch_resampling = 'NoRS'
            switch_reweighting = 'Reweighting'
        elif switch_strategy == 'CB':
            switch_defer = 'NoDefer'
            switch_resampling = 'NoRS'
            switch_reweighting = 'EffectNum'
        elif switch_strategy == 'DRW':
            switch_defer = 'DeferRW'
            switch_resampling = 'NoRS'
            switch_reweighting = 'Reweighting'
        elif switch_strategy == 'DCB':
            switch_defer = 'DeferRW'
            switch_resampling = 'NoRS'
            switch_reweighting = 'EffectNum'

        if i == 0:
            switch_defer0, switch_resampling0, switch_reweighting0 = switch_defer, switch_resampling, switch_reweighting
        elif i == 1:
            switch_defer1, switch_resampling1, switch_reweighting1 = switch_defer, switch_resampling, switch_reweighting
        elif i == 2:
            switch_defer2, switch_resampling2, switch_reweighting2 = switch_defer, switch_resampling, switch_reweighting     

    train_a = ['coarse', 'hier', 'fine', 'pre_hier', 'pre_fine', 'coarse_hier', 'flat'][train_a]
    # the initial code is 'hier', subsequently, 'coarse' and 'fine' are wrote.
    # save_model: coarse;      epoch: 0 ~ coarse_epochoch,                          load_model: pure
    # save_model: hier;        epoch: 0 ~ coarse_epochoch ~ fine_epochoch ~ end_epoch, load_model: pure
    # save_model: fine;        epoch: fine_epochoch ~ end_epoch,                    load_model: pure
    # save_model: pre_hier;    epoch: coarse_epochoch ~ fine_epochoch,                 load_model: coarse
    # save_model: pre_fine;    epoch: fine_epochoch ~ end_epoch,                    load_model: pre_hier
    # save_model: coarse_hier; epoch: 0 ~ coarse_epochoch ~ fine_epochoch,             load_model: pure
    # result: pre_hier ≈ coarse_hier; hier ≈ pre_fine

    end_epoch = coarse_epoch + hier_epoch + fine_epoch
    defer_epoch1 = 0 if switch_defer0 == 'NoDefer' else 160
    defer_epoch2 = coarse_epoch if switch_defer1 == 'NoDefer' else coarse_epoch + 160
    defer_epoch3 = end_epoch - fine_epoch if switch_defer2 == 'NoDefer' else end_epoch - fine_epoch + 160
    batch_size1, batch_size2, batch_size3 = 128, 128, 128
    batch_size_test, shuffle_test = 100, False
    beta_EffectNum_f1, beta_EffectNum_c1, beta_EffectNum_f2, beta_EffectNum_c2, beta_EffectNum_f3, beta_EffectNum_c3 = \
        0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999  # 0.9, 0.99, 0.999, 0.9999
    gamma_Focal1, gamma_Focal2, gamma_Focal3 = 1.0, 1.0, 1.0  # 0.5, 1.0, 1.5
    LDAM_net = True if switch_criterion0 == 'LDAM' or switch_criterion1 == 'LDAM' or \
                       switch_criterion2 == 'LDAM' else False

    import os
    import torch
    import numpy as np

    num_workers = 4  # 4, 8, 16
    pin_memory = True if num_workers != 0 else False
    SEED = SEED if SEED is not None else np.random.randint(10000)
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')

# =====================================================================================================
isTest = False  # isCluster=False, when isTest=False
isCluster = True if isTest else False  # train_a=2;

directory_log0 = os.getcwd().replace('/code', '/log')
directory_log0 = f'{directory_log0}/log/log_{switch_dataset}_{switch_imbalanced}_seed{SEED}/log/'
if isTest:
    directory_test_model = 'log_fine/'
    directory_sw = 'sw00200xf000000_f7065_f200/'
    directory_test_model = directory_log0 + directory_test_model + directory_sw
    test_epoch = 284

    record_test = []
    test_number = 1  #

    directory_log = directory_log0 + f'test/{directory_sw}/'
else:
    if train_a == 'coarse':
        switch = switch.split('h')[0]
        name_switch = name_switch.split('_h_')[0]
        directory_log = directory_log0 + f'log_coarse/sw{switch}/'
    elif train_a == 'hier':
        switch = switch
        name_switch = name_switch
        directory_log = directory_log0 + f'log_hier/sw{switch}/'
    elif train_a == 'fine':
        switch = switch.split('c')[0] + 'f' + switch.split('f')[1]
        name_switch = name_switch.split('_c_')[0] + '_f_' + name_switch.split('_f_')[1]
        directory_log = directory_log0 + f'log_fine/sw{switch}/'
    elif train_a == 'pre_hier':
        load_dir = 'log_coarse'
        load_switch = switch.split('h')[0]
        load_str_epoch = 'final'  # best; final
        load_int_epoch = -1  # -1

        switch = switch.split('f')[0]
        name_switch = name_switch.split('_f_')[0]
        directory_log = directory_log0 + f'log_pre_hier/sw{switch}/'
    elif train_a == 'pre_fine':
        load_dir = 'log_pre_hier'
        # load_dir = 'log_coarse_hier'
        load_switch = switch.split('f')[0]
        load_str_epoch = 'final'  # best; final
        load_int_epoch = -1  # -1

        switch = switch
        name_switch = name_switch
        directory_log = directory_log0 + f'log_pre_fine/sw{switch}/'
    elif train_a == 'coarse_hier':
        switch = switch.split('f')[0]
        name_switch = name_switch.split('_f_')[0]
        directory_log = directory_log0 + f'log_coarse_hier/sw{switch}/'

    if alpha_hier_tail != 0:
        directory_log = directory_log.replace('/sw', f'/alpha{alpha_hier_tail}_beta{beta_semantic_minor}/sw')

    # directory_log = directory_log.rstrip('/') + f'_c{coarse_epoch}h{hier_epoch}f{fine_epoch}/'
    # if os.path.exists(directory_log):
    #     raise ValueError(directory_log)

# =====================================================================================================
system = 0  # require to change the path, when move the entire project among different systmes
if system == 0:
    path_all_dataset = '/home/lzy/Datasets/'
    path_all_dataset_usb = '/media/lzy/大U盘/Datasets/'
    path_custom_tool = '/home/lzy/PycharmProjects/pythonProject/1/'
elif system == 1:
    path_all_dataset = ''
    path_custom_tool = ''

# =====================================================================================================
if __name__ == '__main__':
    import run1