from torch.utils.data import TensorDataset
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import network
import os
import xlwt


def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i = i + 1
    f.save(file_path)  # 保存文件


batch_size = 128
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
trainset = torchvision.datasets.CIFAR10(root=r'./', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=r'./', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

# 测试GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

model = network.resnet18()

# 检测是否存在已经训练好的模型，如果存在的话，读入已训练的模型
if os.path.exists(r'./cifar_net_224.pth'):
    print('[Message] 存在已训练模型，继续训练')
    model.load_state_dict(torch.load(r'./cifar_net_224.pth'))

    # start_epoch = 0 lr =0.001 epoch=100
    # start_epoch=100 lr =0.001 epoch=100
    # start_epoch=200 lr=0.001 epoch=50
    start_epoch = 0
else:
    print('[Message] 不存在已训练模型，训练新模型')
    start_epoch = 0
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fun = torch.nn.CrossEntropyLoss()

datas = []  # 用来存储可视化的数据包括，元素为列表，[epoch,loss,acc]

for epoch in range(100):
    train_loss = 0.0
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    for i, data in enumerate(trainloader):
        x_minibatch, y_minibatch = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        out = model(x_minibatch)
        loss = loss_fun(out, y_minibatch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    torch.save(model.state_dict(), r'./cifar_net_224.pth')
    acc_train = network.evaluate(model, trainloader, device)
    acc = network.evaluate(model, testloader, device)
    print('epoch:%d loss:%f acc_train:%.2f acc_test:%.2f' % (epoch, train_loss, acc_train, acc))
    data = [epoch + 1 + start_epoch, train_loss, acc_train, acc]
    datas.append(data)
    # print("datas:", datas)

data_write(r'./datas_224.xls', datas)
