import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load

from deepinfomax import GcnInfomax

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train(dataset, verbose=True):

    nb_epochs = 3000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    num_layers = 1
    sparse = False

    adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = 2000
    batch_size = 4

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = GcnInfomax(ft_size, hid_units, num_layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0


    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):

        feat_in = torch.FloatTensor(features).cuda()
        adj_in = torch.FloatTensor(adj).cuda()

        model.train()
        optimiser.zero_grad()

        loss = model(feat_in, adj_in)

        loss.backward()

        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        model.eval()

        if sparse:
            adj1 = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff1 = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

        features1 = torch.FloatTensor(features)
        adj1 = torch.FloatTensor(adj)
        diff1 = torch.FloatTensor(diff)
        features1 = features1.cuda()
        adj1 = adj1.cuda()
        diff1 = diff1.cuda()

        embeds1 = model.get_embeddings(features1, diff1)
        embeds2 = model.get_embeddings(features1, adj1)
        embeds = embeds1 + embeds2
        train_embs = embeds[idx_train]
        test_embs = embeds[idx_test]

        train_lbls = labels[idx_train]
        test_lbls = labels[idx_test]

        accs = []
        wd = 0.01 if dataset == 'citeseer' else 0.0

        '''for _ in range(50):
            log = LogReg(hid_units, nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
            log.cuda()
            for _ in range(300):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)

        accs = torch.stack(accs)
        print(accs.mean().item(), accs.std().item())'''

        for _ in range(50):
            classifier = SVC(C=10)

            classifier.fit(train_embs.cpu().numpy(), train_lbls.cpu().numpy())
            accs.append(accuracy_score(test_lbls.cpu().numpy(), classifier.predict(test_embs.cpu().numpy())))

        print(np.mean(accs), np.std(accs))



        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

        #break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    embeds = model.get_embeddings(features, diff)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item())


if __name__ == '__main__':
    #import warnings
    #warnings.filterwarnings("ignore")
    #torch.cuda.set_device(3)

    # 'cora', 'citeseer', 'pubmed'
    dataset = 'cora'
    for __ in range(50):
        train(dataset)


