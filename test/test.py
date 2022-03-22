from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载mnist数据集
# 定义数据转换格式,转化为（1，28*28）
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.resize_(28 * 28))])
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=mnist_transform),
    batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=mnist_transform),
    batch_size=10, shuffle=True)

# 超参数设置
batch_size = 10
epoch = 1
learning_rate = 0.001
# 生成对抗样本的个数
adver_nums = 100


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 选择设备
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# 初始化网络，并定义优化器
simple_model = Net().to(device)
optimizer1 = torch.optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=0.9)
print(simple_model)


# 训练模型
def train(model, optimizer):
    for i in range(epoch):
        for j, (data, target) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            loss = F.cross_entropy(logit, target)
            model.zero_grad()
            # 如下：因为其中的loss是单个tensor就不能用加上一个tensor的维度限制
            loss.backward()
            # 如下有两种你形式表达，一种是原生，一种是使用optim优化函数直接更新参数
            # 为什么原生的训练方式没有效果？？？代表参数没有更新，就离谱。
            # 下面的detach与requires_grad_有讲究哦，终于明白了；但是为什么下面代码不能work还是没搞懂
            # for params in model.parameters():
            #   params = (params - learning_rate * params.grad).detach().requires_grad_()
            optimizer.step()
            if j % 1000 == 0:
                print('第{}个数据，loss值等于{}'.format(j, loss))


train(simple_model, optimizer1)

# 罪魁祸首就是你，没有固定住dropout层的神经元怎么能做测试呢？你模型都么固定，那不是一个输入，会有多种输出值
# 真是被你害死了！！！
simple_model.eval()


# 模型测试
def Test(model, name):
    correct_num = torch.tensor(0).to(device)
    for j, (data, target) in tqdm(enumerate(test_loader)):
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        pred = logit.max(1)[1]
        num = torch.sum(pred == target)
        correct_num = correct_num + num
    print(correct_num)
    print('\n{} correct rate is {}'.format(name, correct_num / 10000))


Test(simple_model, 'simple model')

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# 计算雅可比矩阵，即前向导数
def compute_jacobian(model, input):
    var_input = input.clone()

    var_input.detach_()
    var_input.requires_grad = True
    output = model(var_input)

    num_features = int(np.prod(var_input.shape[1:]))
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


# 计算显著图
def saliency_map(jacobian, target_index, increasing, search_space, nb_features):
    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    # this list blanks out those that are not in the search domain
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
    increase_coef = increase_coef.view(-1, nb_features)

    # calculate sum of target forward derivative of any 2 features.
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features,
                                                                  1)  # PyTorch will automatically extend the dimensions
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
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


def perturbation_single(image, ys_target, theta, gamma, model):
    copy_sample = np.copy(image)
    var_sample = Variable(torch.from_numpy(copy_sample), requires_grad=True).to(device)

    # outputs = model(var_sample)
    # predicted = torch.max(outputs.data, 1)[1]
    # print('测试样本扰动前的预测值：{}'.format(predicted[0]))

    var_target = Variable(torch.LongTensor([ys_target, ])).to(device)

    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = int(np.prod(copy_sample.shape[1:]))
    shape = var_sample.size()

    # perturb two pixels in one iteration, thus max_iters is divided by 2.0
    max_iters = int(np.ceil(num_features * gamma / 2.0))

    # masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it.
    if increasing:
        search_domain = torch.lt(var_sample, 0.99)  # 逐元素比较var_sample和0.99
    else:
        search_domain = torch.gt(var_sample, 0.01)
    search_domain = search_domain.view(num_features)

    model.eval().to(device)
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].cpu().numpy()

    iter = 0
    while (iter < max_iters) and (current[0] != ys_target) and (search_domain.sum() != 0):
        # calculate Jacobian matrix of forward derivative
        jacobian = compute_jacobian(model, var_sample)
        # get the saliency map and calculate the two pixels that have the greatest influence
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        # apply modifications
        var_sample_flatten = var_sample.view(-1, num_features).clone().detach_()
        var_sample_flatten[0, p1] += theta
        var_sample_flatten[0, p2] += theta

        new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
        new_sample = new_sample.view(shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        var_sample = Variable(torch.tensor(new_sample), requires_grad=True).to(device)

        output = model(var_sample)
        current = torch.max(output.data, 1)[1].cpu().numpy()
        iter += 1

    adv_samples = var_sample.data.cpu().numpy()
    return adv_samples


# 这几个变量主要用于之后的测试以及可视化
adver_example_by_JSMA = torch.zeros((batch_size, 1, 28, 28)).to(device)
adver_target = torch.zeros(batch_size).to(device)
clean_example = torch.zeros((batch_size, 1, 28, 28)).to(device)
clean_target = torch.zeros(batch_size).to(device)

theta = 1.0  # 扰动值
gamma = 0.1  # 最多扰动特征数占总特征数量的比例
ys_target = 2  # 对抗性样本的标签
# 从test_loader中选取1000个干净样本，使用deepfool来生成对抗样本
for i, (data, target) in enumerate(test_loader):
    if i >= adver_nums / batch_size:
        break
    if i == 0:
        clean_example = data
    else:
        clean_example = torch.cat((clean_example, data), dim=0)

    cur_adver_example_by_JSMA = torch.zeros_like(data).to(device)

    for j in range(batch_size):
        # image = testdata[index][0].resize_(1,784).numpy() # 测试样本特征
        # data1 = data[j]
        # print (data.shape)
        pert_image = perturbation_single(data[j].resize_(1, 28 * 28).numpy(), ys_target, theta, gamma, simple_model)
        # print (1)
        cur_adver_example_by_JSMA[j] = torch.from_numpy(pert_image).to(device)

    # 使用对抗样本攻击VGG模型
    pred = simple_model(cur_adver_example_by_JSMA).max(1)[1]
    if i == 0:
        adver_example_by_JSMA = cur_adver_example_by_JSMA
        clean_target = target
        adver_target = pred
    else:
        adver_example_by_JSMA = torch.cat((adver_example_by_JSMA, cur_adver_example_by_JSMA), dim=0)
        clean_target = torch.cat((clean_target, target), dim=0)
        adver_target = torch.cat((adver_target, pred), dim=0)

print(adver_example_by_JSMA.shape)
# print (adver_target)
print(clean_example.shape)


# print (clean_target)

def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target):
    n_cols = 5
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(n_cols * 4, n_rows * 2))
    for i in range(n_cols):
        for j in range(n_rows):
            plt.subplot(n_cols, n_rows * 2, cnt1)
            plt.xticks([])
            plt.yticks([])
            plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(clean_example[cnt].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
            plt.subplot(n_cols, n_rows * 2, cnt1 + 1)
            plt.xticks([])
            plt.yticks([])
            # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(adver_example[cnt].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
            cnt = cnt + 1
            cnt1 = cnt1 + 2
    plt.show()


plot_clean_and_adver(adver_example_by_JSMA, adver_target, clean_example, clean_target)
