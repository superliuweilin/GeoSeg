# 读取数据
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv("/home/lyu3/lwl_wp/lightning_logs/vaihingen/unetformer-r18-512-crop-ms-e100/version_0/metrics.csv")
print(data)
# data1 = data.iloc[4, :]
# print(data1)
# 绘制每一轮损失的变化图
Loss_list = data['epoch']  # 存储每次epoch损失值
print(Loss_list)
# def draw_loss(Loss_list, epoch):
#             # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
#             plt.cla()
#             x1 = range(1, epoch + 1)
#             print(x1)
#             y1 = Loss_list
#             print(y1)
#             plt.title('Train loss vs. epoches', fontsize=20)
#             plt.plot(x1, y1, '.-')
#             plt.xlabel('epoches', fontsize=20)
#             plt.ylabel('Train loss', fontsize=20)
#             plt.grid()
#             plt.savefig("./loss/Train_loss.png")
#             plt.show()

