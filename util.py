import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
import pandas as pd
import csv

def myprint(filename, inform):
    print(inform)
    f = open(filename, 'a')
    f.write(str(inform) + '\n')
    f.close()

def count_params(_vars=None, scope=None):
    total_params = 0
    if scope == None: scope = 'Whole Model'
    if _vars == None: _vars = tf.trainable_variables()
    for variable in _vars:
        shape = variable.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim
        total_params += var_params
    print("Number of trainable params for {} : {}".format(scope, total_params))

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))

def write(csvfile, audioname, perm, epoch, n_speaker):
    f = open(csvfile, 'w')
    w = csv.writer(f)
    header = ['audio']
    for i in range(epoch):
        header.append('epoch'+str(i+1))
    w.writerow(header)

    perm_dict = {audioname[k]:v for k,v in perm.items()}

    for key, value in perm_dict.items():
        w.writerow([key] + [i[0] for i in value])
        for spk in range(1, n_speaker):
            w.writerow([''] +  [i[spk] for i in value])
    f.close()

def load_perm(config, mode, g_global_step, dataset, size, _type='perm'):
    data = pd.read_csv(config['training']['path']+'/'+str(mode)+'_'+_type+'.csv')
    perm = []
    for e in range(1, g_global_step*config['training']['batch_size']//20000+1):
        perm.append(data['epoch'+str(e)].values.reshape(-1,2))
    utt_idx = [data['audio'][2*i] for i in range(size)]
    myorder = []
    for i in range(size):
        myorder.append(utt_idx.index(dataset.file_base[i]))
    perm = np.transpose(np.array(perm), (1,0,2))
    adjust_perm = [perm[i] for i in myorder]

    return {i: list(adjust_perm[i]) for i in range(size) }

def write_pretrained_perm(model, epoch):
    data = pd.read_csv(os.path.join('models', model, 'tr_perm.csv'))
    perm = data['epoch'+str(epoch)].values.reshape(-1,2)

    if not os.path.exists(os.path.join('models', model, 'perm_idx')):
        os.mkdir(os.path.join('models', model, 'perm_idx'))

    np.savetxt(os.path.join('models', model, 'perm_idx', str(epoch)+'.csv'), perm, fmt='%i', delimiter=',')