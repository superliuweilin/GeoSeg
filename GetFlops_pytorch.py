import torch
import time
from geoseg.models.CSAvitNet import CSAvitNet

# 加载你的神经网络模型
model = CSAvitNet()

# 准备测试用的输入数据，例如：
input_data = torch.randn(1, 3, 512, 512)  # 示例输入数据
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
input_data = input_data.to(device)
# 运行模型推理，并测量每秒处理的帧数
with torch.no_grad():
    warmup_iterations = 100  # 模型预热迭代次数

    # 模型预热
    for _ in range(warmup_iterations):
        output = model(input_data)

    total_time = 0
    num_iterations = 300  # 迭代次数，用于测量帧率
    for _ in range(num_iterations):
        start_time = time.time()

        # 执行模型推理
        output = model(input_data)

        end_time = time.time()
        total_time += end_time - start_time

    # 计算帧率
    fps = num_iterations / total_time
    print("FPS: {:.2f}".format(fps))

