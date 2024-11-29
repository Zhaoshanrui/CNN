# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:48:49 2024

@author: DELL
"""

#加载包
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

#######################################数据预处理############################################
train_data = pd.read_csv("D:/myfiles/yanyi_kaggle/cug-24-fallasm/train.csv")
test_data = pd.read_csv("D:/myfiles/yanyi_kaggle/cug-24-fallasm/test.csv")
train_data = train_data.fillna(train_data.mean())
train_data_array = np.array(train_data)
test_data = test_data.fillna(test_data.mean())
test_data_array = np.array(test_data)

######################################划分x，y###############################################
b = train_data_array[:, 21]
a = train_data_array[:, 0:21]

a_test = test_data_array[:, 0:21] #待预测数据

###################################划分训练集测试集##########################################
x_train, x_test, y_train, y_test = train_test_split(a, b, test_size = 0.3, random_state = 1)

####################################卷积神经网络（CNN）######################################

## 数据归一化处理
scaler_x = MinMaxScaler()
# 创建一个MinMaxScaler对象scaler_x,用于对输入特征X进行归一化处理
# MinMaxScaler会将数据缩放到[0, 1]的范围内

X = scaler_x.fit_transform(a)
# 使用scaler_x对输入特征a进行拟合和转换
# fit_transform方法会计算数据的最小值和最大值,并将数据缩放到[0, 1]的范围内
# 转换后的数据将覆盖原始的X

scaler_y = MinMaxScaler()
# 创建另一个MinMaxScaler对象scaler_y,用于对目标值y进行归一化处理

y = scaler_y.fit_transform(b.reshape(-1, 1)).flatten()
# 使用scaler_y对目标值y进行拟合和转换
# 由于MinMaxScaler期望输入是二维数组,因此需要使用reshape(-1, 1)将y转换为二维数组
# reshape(-1, 1)表示将y转换为一个列向量,行数自动推断
# fit_transform方法会计算数据的最小值和最大值,并将数据缩放到[0, 1]的范围内
# 最后使用flatten()将转换后的二维数组重新转换为一维数组,覆盖原始的y

## 将数据转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
# 使用torch.tensor函数将NumPy数组X转换为PyTorch张量
# dtype=torch.float32指定张量的数据类型为32位浮点数
# 转换后的张量X将用于模型的输入
 
y = torch.tensor(y, dtype=torch.float32)
# 转换后的张量y将用于模型的训练和评估

## 数据集划分
train_ratio = 0.7
val_ratio = 0.1
# 定义训练集和验证集的比例
# train_ratio表示训练集占总数据的比例,这里设置为0.7,即70%的数据用于训练
# val_ratio表示验证集占总数据的比例,这里设置为0.1,即10%的数据用于验证
# 剩下的20%用于测试
 
num_samples = len(X)
# 获取数据集的样本数,即X的长度
 
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
# 计算训练集和验证集的样本数
# num_train表示训练集的样本数,通过总样本数乘以训练集比例并取整得到
# num_val表示验证集的样本数,通过总样本数乘以验证集比例并取整得到
 
train_data = X[:num_train]
train_labels = y[:num_train]
# 使用切片操作提取训练集数据和标签
# train_data表示训练集的输入特征,取X的前num_train个样本
# train_labels表示训练集的目标值,取y的前num_train个样本
 
val_data = X[num_train:num_train+num_val]
val_labels = y[num_train:num_train+num_val]
# 使用切片操作提取验证集数据和标签
# val_data表示验证集的输入特征,取X从num_train到num_train+num_val的样本
# val_labels表示验证集的目标值,取y从num_train到num_train+num_val的样本
 
test_data = X[num_train+num_val:]
test_labels = y[num_train+num_val:]
# 使用切片操作提取测试集数据和标签
# test_data表示测试集的输入特征,取X从num_train+num_val到最后的样本
# test_labels表示测试集的目标值,取y从num_train+num_val到最后的样本

## 创建数据加载器
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)
# 使用TensorDataset将训练集、验证集和测试集的数据和标签打包成数据集对象
# TensorDataset接受多个张量作为参数,将它们组合成一个数据集
# train_dataset表示训练集的数据集对象,包含训练数据和标签
# val_dataset表示验证集的数据集对象,包含验证数据和标签
# test_dataset表示测试集的数据集对象,包含测试数据和标签
 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
# 使用DataLoader创建数据加载器,用于批次读取数据
# train_loader表示训练集的数据加载器,batch_size=64表示每个批次包含64个样本,shuffle=True表示在每个epoch开始时打乱数据顺序
# val_loader表示验证集的数据加载器,batch_size=64表示每个批次包含64个样本
# test_loader表示测试集的数据加载器,batch_size=64表示每个批次包含64个样本

## 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(2, 2))
        # 定义一个二维卷积层,输入通道为1,输出通道为32,卷积核大小为(3, 1),padding为(2, 2)
        self.relu1 = nn.ReLU()
        # 定义一个ReLU激活函数
        
        # 计算卷积层输出的维度
        conv_output_dim = self.conv1(torch.zeros(1, 1, input_dim, 1)).view(-1).shape[0]
        # 通过将全零张量传递给卷积层并展平输出,计算卷积层输出的维度
        self.fc1 = nn.Linear(conv_output_dim, 1)
        # 定义一个全连接层,输入维度为卷积层输出的维度,输出维度为1
 
    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(3)  # 增加通道维度和高度维度
        # 在输入张量x上增加通道维度和高度维度,以满足卷积层的输入要求
        x = self.conv1(x)
        # 将输入张量x传递给卷积层
        x = self.relu1(x)
        # 对卷积层的输出应用ReLU激活函数
        x = x.view(x.size(0), -1)
        # 将卷积层的输出展平为二维张量,第一维为批次大小,第二维为特征维度
        x = self.fc1(x)
        # 将展平后的张量传递给全连接层,得到最终的输出
        return x
 
model = CNN(input_dim=X.shape[1])
# 创建CNN模型的实例,输入维度为X的特征数

## 设置损失函数和优化器
criterion = nn.MSELoss()
# 定义均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 定义Adam优化器,学习率为0.01,优化对象为模型的参数

## 训练模型
num_epochs = 20
# 设置训练的轮数为20
train_losses = []
val_losses = []
# 定义用于存储训练损失和验证损失的列表
 
for epoch in range(num_epochs):
    model.train()
    # 将模型设置为训练模式
    train_loss = 0.0
    # 初始化训练损失为0.0
    for data, labels in train_loader:
        # 遍历训练数据加载器,获取每个批次的数据和标签
        optimizer.zero_grad()
        # 将优化器的梯度置零
        outputs = model(data)
        # 将数据输入模型,得到预测输出
        loss = criterion(outputs, labels.unsqueeze(1))
        # 计算预测输出和真实标签之间的损失,需要将标签增加一个维度以匹配输出的形状
        loss.backward()
        # 反向传播计算梯度
        optimizer.step()
        # 更新模型参数
        train_loss += loss.item() * data.size(0)
        # 累加训练损失,乘以批次大小以得到总损失
    train_loss /= len(train_loader.dataset)
    # 计算平均训练损失,除以训练集的样本数
    train_losses.append(train_loss)
    # 将平均训练损失添加到训练损失列表中
 
    model.eval()
    # 将模型设置为评估模式
    val_loss = 0.0
    # 初始化验证损失为0.0
    with torch.no_grad():
        # 禁用梯度计算,以减少内存占用和加速计算
        for data, labels in val_loader:
            # 遍历验证数据加载器,获取每个批次的数据和标签
            outputs = model(data)
            # 将数据输入模型,得到预测输出
            loss = criterion(outputs, labels.unsqueeze(1))
            # 计算预测输出和真实标签之间的损失,需要将标签增加一个维度以匹配输出的形状
            val_loss += loss.item() * data.size(0)
            # 累加验证损失,乘以批次大小以得到总损失
    val_loss /= len(val_loader.dataset)
    # 计算平均验证损失,除以验证集的样本数
    val_losses.append(val_loss)
    # 将平均验证损失添加到验证损失列表中
 
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    # 打印当前轮数、训练损失和验证损失

# 在测试集上评估模型
model.eval()
# 将模型设置为评估模式
test_preds = []
# 定义用于存储测试集预测值的列表
with torch.no_grad():
    # 禁用梯度计算,以减少内存占用和加速计算
    for data, _ in test_loader:
        # 遍历测试数据加载器,获取每个批次的数据
        outputs = model(data)
        # 将数据输入模型,得到预测输出
        test_preds.extend(outputs.numpy())
        # 将预测输出转换为NumPy数组并添加到测试集预测值列表中
 
test_preds = scaler_y.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
# 对测试集预测值进行反归一化,将其转换为原始尺度
test_labels = scaler_y.inverse_transform(test_labels.numpy().reshape(-1, 1)).flatten()
# 对测试集真实标签进行反归一化,将其转换为原始尺度

score = explained_variance_score(test_labels, test_preds)
print("准确率为：", score)

## 绘制测试集的真实值和预测值
plt.figure(figsize=(8, 6))
# 创建一个大小为(8, 6)的图形
plt.plot(test_labels, label='True Values (Testing Set)')
# 绘制测试集的真实标签,并添加标签
plt.plot(test_preds, label='Predicted Values (Testing Set)')
# 绘制测试集的预测值,并添加标签
plt.xlabel('Sample')
# 设置x轴标签为"Sample"
plt.ylabel('House Price')
# 设置y轴标签为"House Price"
plt.title('True vs. Predicted Values (Testing Set)')
# 设置图形标题为"True vs. Predicted Values (Testing Set)"
plt.legend()
# 添加图例
plt.tight_layout()
# 调整子图参数,使之填充整个图像区域
plt.show()
# 显示图形