import numpy as np
from tensorboardX import SummaryWriter
import xlrd

worksheet = xlrd.open_workbook(r'./datas_224.xls')
sheet1 = worksheet.sheet_by_name('sheet1')

epoch = []
train_loss = []
train_acc = []
acc = []
# print(sheet1.nrows)
for i in range(sheet1.nrows):
    epoch.append(sheet1.cell(i, 0).value)
    train_loss.append(sheet1.cell(i, 1).value)
    train_acc.append(sheet1.cell(i, 2).value)
    acc.append(sheet1.cell(i, 3).value)

# 测试查看导入结果
# print("epoch:", epoch[0:4])
# print("train loss:", train_loss[0:4])
# print("acc:", acc[0:4])

# 转换为ndarray数组
# epoch = np.ndarray(epoch)
# train_loss = np.ndarray(train_loss)
# acc = np.ndarray(acc)

writer = SummaryWriter()
for i in range(sheet1.nrows):
    writer.add_scalar('scalar/loss_224', train_loss[i], epoch[i])
    writer.add_scalar('scalar/test_acc_224', acc[i], epoch[i])
    writer.add_scalar('scalar/train_acc_224', train_acc[i], epoch[i])
writer.close()
