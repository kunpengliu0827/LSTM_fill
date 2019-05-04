import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_fill = pd.read_csv('1.csv')[['Hours', 'Respiratory rate']].values
df = pd.read_csv('1.csv')[['Hours', 'Respiratory rate']]
print(df)
'''
判断某一个数据是否为空，如果为空则保存它的时间戳
'''
nan_data = []
index = 0
for x in df.values[:, 1]:
    if np.isnan(x):
        nan_data.append(df.values[:, 0][index])
    index += 1
nan_data = np.array(nan_data)

df.dropna(axis=0, how='any', inplace=True)
new_data = df.values
X = new_data[:, 0].reshape(-1, 1, 1)
y = new_data[:, 1].reshape(-1, 1, 1)


import torch
from torch import nn
from torch.autograd import Variable


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape  # (seq, batch, hidden)
        x = x.view(s * b, h)  # 转化为线性层的输入方式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


net = lstm_reg(1, 32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(torch.from_numpy(X).float())
    var_y = Variable(torch.from_numpy(y).float())
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e + 1, loss.item()))

# 训练完成之后，我们可以用训练好的模型去预测后面的结果
net = net.eval()
data_X = nan_data.reshape(-1, 1, 1)
data_X = torch.from_numpy(data_X).float()
var_data = Variable(data_X)
pred_test = net(var_data)  # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()


for index in range(0, len(df_fill)):
    for f in range(0, len(nan_data)):
        if df_fill[index][0] == nan_data[f]:
            df_fill[index][1] = pred_test[f]


# 画出实际结果和预测的结果
plt.plot(df_fill[:, 1], 'r', label='prediction')
plt.plot(new_data[:, 1], 'b', label='real')
plt.legend(loc='best')
plt.show()



