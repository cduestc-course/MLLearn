import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 1. 定义二维 Sigmoid 函数
def sigmoid_2d(X, Y):
    # 可以使用 X + Y 作为输入，或者其他线性组合，例如：a*X + b*Y
    Z = 1 / (1 + np.exp(-(X + Y)))
    return Z

# 2. 创建数据点
# 在 X 和 Y 方向上创建等间距的点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# 使用 np.meshgrid 创建 X 和 Y 的网格数据
X, Y = np.meshgrid(x, y)

# 计算 Z 坐标（Sigmoid 函数值）
Z = sigmoid_2d(X, Y)

# 3. 绘制三维曲面图
fig = plt.figure(figsize=(10, 8))
# 添加 3D 坐标轴
ax = fig.add_subplot(111, projection='3d')

# 使用 plot_surface 绘制曲面
# cmap='viridis' 是一个常用的颜色映射
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                    linewidth=0, antialiased=False)

# 4. 设置图形标题和标签
ax.set_title('Sigmoid Function 3D Surface Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis ($\sigma(X+Y)$)')

# 设置 Z 轴的显示范围，Sigmoid 函数的值域是 (0, 1)
ax.set_zlim(0.0, 1.0)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5, label='Sigmoid Value')

# 显示图形
plt.show()