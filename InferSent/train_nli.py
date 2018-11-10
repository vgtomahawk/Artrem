# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--extNlipath",type=str,default='dataset/HardNLI/', help="external NLI Path")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--no_early_stopping",action='store_true',help="don't cut the model training after two validation decreases")
parser.add_argument("--evaluateOnly",action="store_true",help="just evaluate, don't train")
parser.add_argument("--evalExt",action="store_true",help="just evaluate, don't train")


# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# adversary parameters
parser.add_argument("--use_adv",action='store_true',help="whether to use the adversary loss or not")
parser.add_argument("--lambda_adv",type=float,default=0.001,help="coefficient of the adversarial loss")
parser.add_argument("--deeper_adv",action='store_true',help="deeper adversary")
parser.add_argument("--separate_encoder",action="store_true",help="separate encoder")
parser.add_argument("--full_through_adversary",action='store_true',help="full_through_adversary")
parser.add_argument("--reversal_weight",type=float,default=1.0,help="reversal_weight")

#adversary annealing parameters
parser.add_argument("--annealing",action='store_true',help="anneal in adversary loss weight")
parser.add_argument("--addAnnealing",action='store_true',help="adding annealing")
parser.add_argument("--max_lambda_adv",type=float,default=0.01,help="anneal to this value maximum")
parser.add_argument("--anneal_growth_rate",type=float,default=1.1,help="grow lambda_adv by this value every epoch. Must be >1")


# gpu
parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
if params.gpu_id >= 0:
    torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
if params.gpu_id >= 0:
    torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

if params.evalExt:
    _, extValid, extTest = get_nli(params.extNlipath)
    for split in ['s1', 's2']:
        for data_type in ['extValid','extTest']:
            eval(data_type)[split] = np.array([['<s>'] +
                [word for word in sent.split() if word in word_vec] +
                ['</s>'] for sent in eval(data_type)[split]])



"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  False                  ,
    'use_adv'        :  params.use_adv         , 
    'deeper_adv'     :  params.deeper_adv      ,
    'reversal_weight' : params.reversal_weight ,
    'full_through_adversary': params.full_through_adversary ,
    'separate_encoder': params.separate_encoder,
}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
if params.gpu_id >= 0:
    nli_net.cuda()
    loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    if params.use_adv: adverseCorrect = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    if params.addAnnealing: numberOfIters = (len(s1)+0.0)/(params.batch_size+0.0)
    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size]))
        if params.gpu_id >= 0:
            s1_batch = s1_batch.cuda()
            s2_batch = s2_batch.cuda()
            tgt_batch = tgt_batch.cuda()

        k = s1_batch.size(1)  # actual batch size

        # model forward
        output, adversaryOutput = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])
        
        if params.use_adv:
            adversePred = adversaryOutput.data.max(1)[1]
            adverseCorrect += adversePred.long().eq(tgt_batch.data.long()).cpu().sum()
            assert len(adversePred) == len(s1[stidx:stidx + params.batch_size])
           
        # loss
        loss = loss_fn(output, tgt_batch)
        if params.use_adv:
            adversaryLoss = loss_fn(adversaryOutput,tgt_batch)
            currentLambda = params.lambda_adv
            if params.annealing:
                currentLambda = min(params.max_lambda_adv, currentLambda*(params.anneal_growth_rate**epoch) )
            if params.addAnnealing:
                currentLambda = min(params.max_lambda_adv, currentLambda + (epoch*numberOfIters+stidx/params.batch_size)*params.anneal_growth_rate)
            loss = loss + currentLambda*adversaryLoss
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            #import pdb;pdb.set_trace()
            if params.use_adv:
                logs.append('{0} ; loss {1:.2f} ; sentence/s {2} ; words/s {3} ; accuracy train : {4:.4f} ; lambda : {5:.4f} adversary accuracy train: {6:.4f} '.format(
                            stidx, np.mean(all_costs),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            100.*(correct.item())/(stidx+k+0.0),
                            currentLambda,
                            100.*(adverseCorrect.item())/(stidx+k+0.0) ))
            else:
                logs.append('{0} ; loss {1:.2f} ; sentence/s {2} ; words/s {3} ; accuracy train : {4:.4f}'.format(
                            stidx, np.mean(all_costs),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            100.*(correct.item())/(stidx+k+0.0) ))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = 100 * (correct+0.0)/(len(s1)+0.0)
    print('results : epoch {0} ; mean accuracy train : {1:.5f}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    if eval_type == 'valid' or eval_type=='test':
        s1 = valid['s1'] if eval_type == 'valid' else test['s1']
        s2 = valid['s2'] if eval_type == 'valid' else test['s2']
        target = valid['label'] if eval_type == 'valid' else test['label']
    elif eval_type == 'extValid':
        s1,s2,target = extValid['s1'], extValid['s2'], extValid['label']
    else:
        s1,s2,target = extTest['s1'], extTest['s2'], extTest['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])) #.cuda()
        if params.gpu_id >= 0:
            s1_batch = s1_batch.cuda()
            s2_batch = s2_batch.cuda()
            tgt_batch = tgt_batch.cuda()

        # model forward
        output, adversaryOutput = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = 100 * (correct.item()) / (len(s1)+0.0)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1


# Load the best model so far.
#nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

if not params.evaluateOnly:
    while (not stop_training or params.no_early_stopping) and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)
        eval_acc = evaluate(epoch, 'valid')
        epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

if params.evalExt:
    evaluate(1e6, 'extValid', True)
    evaluate(0, 'extTest', True)

# Save encoder instead of full model
#torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
