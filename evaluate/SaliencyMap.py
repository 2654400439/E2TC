import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from TargetModel.FSNet.train import computeFPR
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from TargetModel.FSNet.dataset import C2Data
from substituteModel.DNN import trainMTa
from advertorch.attacks import JacobianSaliencyMapAttack, GradientSignAttack, CarliniWagnerL2Attack, LinfBasicIterativeAttack

class CICIDS2017_Statistic(Dataset):
    def __init__(self, name):
        self.data = np.load("/home/sunhanwu/datasets/cicids2017/npy/{}.npy".format(name), allow_pickle=True)
        X = self.data[:, :-1].astype(np.float32)
        self.data[self.data[:, -1]!= 0] = 1
        y = self.data[:, -1].astype(int)
        X[np.where(np.isnan(X))] = 0
        X[np.where(X >= np.finfo(np.float32).max)] = np.finfo(np.float32).max - 1
        X = normalize(X, axis=0, norm='max')
        self.X = X
        self.y = y


    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.data)

class CICIDS2017_Sequence(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

class MTA_Statistic(Dataset):
    def __init__(self, name):
        self.data = np.load("/home/sunhanwu/datasets/MTA/cicflownpy/{}.npy".format(name), allow_pickle=True)
        X = self.data[:, :-1].astype(np.float32)
        self.data[self.data[:, -1]!= 0] = 1
        y = self.data[:, -1].astype(int)
        X[np.where(np.isnan(X))] = 0
        X[np.where(X >= np.finfo(np.float32).max)] = np.finfo(np.float32).max - 1
        X = normalize(X, axis=0, norm='max')
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.data)

class MTA_Sequence(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

class TargetDNN(nn.Module):
    """
    target DNN model
    """
    def __init__(self, param):
        """

        :param param:
        """
        super(TargetDNN, self).__init__()
        self.input_size = param['input_size']
        self.class_num = param['num_class']

        self.linear1 = nn.Linear(self.input_size, int(self.input_size / 2))
        self.linear2 = nn.Linear(int(self.input_size / 2), int(self.input_size / 4))
        self.linear3 = nn.Linear(int(self.input_size / 4), self.class_num)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x
def trainDNN(dataset, name, param):
    print("train DNN model for {} ".format(name))
    # hyper param
    epoch_size=10
    batch_size = 32
    lr = 1e-3
    num = len(dataset)

    test_size = int(num * 0.1)
    train_size = int((num - test_size) * 0.9)
    valid_size = num - test_size - train_size

    # use GPU if it is available, oterwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_valid_data, test_data = torch.utils.data.random_split(dataset, [train_size + valid_size, test_size])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    # model
    dnn = TargetDNN(param)
    dnn.to(device)

    # loss func and optim
    crossEntropy = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(dnn.parameters(), lr=lr)
    for i in range(epoch_size):
        dnn.train()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in tqdm(train_loader):
            batch_x = batch_x.to(device, dtype=torch.float)
            batch_y = batch_y.to(device)
            output = dnn(batch_x)
            # output.shape = (batch_size, sequence, num_class)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            # ipdb.set_trace()
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            output = F.softmax(output)
            loss = crossEntropy(output, batch_y)

            acc_list.append(acc)
            recall_list.append(recall)
            f1_list.append(f1)
            loss_list.append(loss.item())

            adam.zero_grad()
            loss.backward()
            adam.step()
        print("[Training {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                               np.mean(acc_list),
                                                                                               np.mean(recall_list),
                                                                                               np.mean(f1_list),
                                                                                               np.mean(loss_list)))

        # validing
        dnn.eval()
        loss_list = []
        acc_list = []
        recall_list = []
        f1_list = []
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device, dtype=torch.float)
            batch_y = batch_y.to(device)
            output = dnn(batch_x)
            # output.shape = (batch_size, sequence, num_class)
            acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
            batch_y = batch_y.squeeze()
            # batch_y = F.softmax(batch_y)
            # output = F.softmax(output)
            loss = crossEntropy(output, batch_y)

            acc_list.append(acc)
            recall_list.append(recall)
            f1_list.append(f1)
            loss_list.append(loss.item())
        print("[Validing {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                               np.mean(acc_list),
                                                                                               np.mean(recall_list),
                                                                                               np.mean(f1_list),
                                                                                               np.mean(loss_list)))

    # testing
    dnn.eval()
    loss_list = []
    acc_list = []
    recall_list = []
    f1_list = []
    y_true = []
    y_pred = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, dtype=torch.float)
        batch_y = batch_y.to(device)
        output = dnn(batch_x)
        # output.shape = (batch_size, sequence, num_class)
        acc, recall, f1 = computeFPR(y_pred=output, y_target=batch_y)
        batch_y = batch_y.squeeze()
        # batch_y = F.softmax(batch_y)
        # output = F.softmax(output)
        loss = crossEntropy(output, batch_y)

        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1)
        loss_list.append(loss.item())
        y_true += batch_y.detach().cpu().numpy().tolist()
        y_pred += torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
    print("[Testing {:03d}] acc: {:.2%}, recall: {:.2%}, f1: {:.2%}, loss: {:.2f}".format(i + 1,
                                                                                          np.mean(acc_list),
                                                                                          np.mean(recall_list),
                                                                                          np.mean(f1_list),
                                                                                          np.mean(loss_list)))
    print(confusion_matrix(y_true, y_pred))
    return dnn

# def jsmaAttack(model: nn.Module, num_class: int, dataloader: DataLoader, device, target_class: int,
#                clip_min=0.0, clip_max=1.0, theta=1.0, gamma=1.0, modify_times=10,
#                filter_array: list =None):
#     """
#     :return:
#     """
#     model = model.to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     adv_dataset = []
#     jsma = JacobianSaliencyMapAttack(
#         model, num_class, clip_min=-1, clip_max=1, loss_fn=loss_fn, theta=theta, gamma=gamma, modify_times=modify_times)
#     batch_x_clone = None
#     adv_x = None
#     origin_dataset = []
#     for batch_x, batch_y in tqdm(dataloader):
#         batch_x = batch_x.float().to(device)
#         batch_x_clone = batch_x.clone()
#         batch_y = batch_y.to(device)
#         target = torch.from_numpy(np.array([target_class] * batch_x.shape[0]))
#         target = target.to(device)
#         adv_x = jsma.perturb(batch_x, target)
#         if filter_array:
#             filter = torch.tensor(filter_array)
#             filter = filter.repeat((batch_x.shape[0], 1))
#             adv_x[filter == 0] = -0xffffffffffff
#             batch_x_clone[filter == 1] = -0xffffffffffff
#             adv_x = torch.max(adv_x, batch_x_clone)
def compute_jacobian(model, X):
    var_input = X.clone()
    var_input.detach_()
    var_input.requires_grad = True
    output = model(var_input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = int(np.prod(X.shape[1:]))
    jacobian = torch.zeros([output.size()[1], num_features])
    for i in range(output.size()[1]):
        # zero_gradients(input)
        if var_input.grad is not None:
            var_input.grad.zero_()
        # output.backward(mask,retain_graph=True)
        output[0][i].backward(retain_graph=True)
        # copy the derivative to the target place
        jacobian[i] = var_input.grad.squeeze().view(-1, num_features).clone()

    return jacobian.to(device)

def saliency_map(jacobian, target_index, increasing, search_space, nb_features):
    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # this list blanks out those that are not in the search domain
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
    increase_coef = increase_coef.view(-1, nb_features)

    # calculate sum of target forward derivative of any 2 features.
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  # PyTorch will automatically extend the dimensions
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # zero out the situation where a feature sums with itself
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte().to(device)

    # According to the definition of saliency map in the paper (formulas 8 and 9),
    # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    # apply the mask to the saliency map
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    # do the multiplication according to formula 10 in the paper
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    # get the most significant two pixels
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    map = torch.max(saliency_map, dim=1)[0]
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q, map

def perturbation_single(X, ys_target, theta, gamma, model, device):

    X = X.to(device)
    target_y = torch.LongTensor([ys_target]).to(device)
    if theta > 0:
        increasing = True
    else:
        increasing = False
    num_features = int(np.prod(X.shape[1:]))
    if increasing:
        search_domain = torch.lt(X, 0.99) #逐元素比较var_sample和0.99
    else:
        search_domain = torch.gt(X, 0.01)
    search_domain = search_domain.view(num_features)

    model.eval().to(device)
    jacobian = compute_jacobian(model, X)
    p1, p2, map = saliency_map(jacobian, target_y, increasing, search_domain, num_features)

    return p1.item(), p2.item(), map.detach_().cpu().tolist()

def run_CICIDS2017_Statistic():
    malwareFamily = ['Botnet', 'BruteForce', 'DDoS', 'Fuzzing', 'PortScan']
    param = {
        "input_size": 77,
        "num_class": 2
    }
    for cls in malwareFamily:
        cicids_data = CICIDS2017_Statistic(cls)
        model = trainDNN(cicids_data, cls, param)
        dataloader = DataLoader(cicids_data, batch_size=1, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_list = []
        for batch_x, batch_y in dataloader:
            if batch_y.item() == 0:
                continue
            p, q, map= perturbation_single(batch_x, 0, theta=0.5, gamma=1.0, model=model, device=device)
            if p != 0 or q != 0:
                map_list.append(map)
        map_list = np.array(map_list)
        np.save("/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/map_{}.npy".format(cls), map_list)


def run_MTA_Statistic():
    malwareFamily = ['Dridex', 'Gozi', 'Quakbot', 'Tofsee', 'TrickBot']
    param = {
        "input_size": 77,
        "num_class": 2
    }
    for cls in malwareFamily:
        mta_data = MTA_Statistic(cls)
        model = trainDNN(mta_data, cls, param)
        dataloader = DataLoader(mta_data, batch_size=1, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_list = []
        for batch_x, batch_y in dataloader:
            if batch_y.item() == 0:
                continue
            p, q, map= perturbation_single(batch_x, 0, theta=0.5, gamma=1.0, model=model, device=device)
            if p != 0 or q != 0:
                map_list.append(map)
        map_list = np.array(map_list)
        np.save("/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/map_{}.npy".format(cls), map_list)

def run_MTA_Sequence():
    malwareFamily = {'Dridex': 8000,
                     'Gozi':580,
                     'Quakbot':700,
                     'Tofsee':10000,
                     'TrickBot':650
                     }
    param = {
        "input_size": 80,
        "num_class": 2
    }
    for cls in malwareFamily.keys():
        # mta_data = MTA_Statistic(cls)
        mta_data = C2Data(cls, number=malwareFamily[cls], sequenceLen=80)
        model = trainMTa(cls, 'dnn', sample_size=malwareFamily[cls], feature_type='length')
        dataloader = DataLoader(mta_data, batch_size=1, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_list = []
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float()
            if batch_y.item() == 0:
                continue
            p, q, map= perturbation_single(batch_x, 0, theta=0.5, gamma=1.0, model=model, device=device)
            if p != 0 or q != 0:
                map_list.append(map)
        map_list = np.array(map_list)
        np.save("/home/sunhanwu/work2021/TrafficAdversarial/experiment/src/result/map_{}.npy".format(cls), map_list)
if __name__ == '__main__':
    run_MTA_Sequence()
