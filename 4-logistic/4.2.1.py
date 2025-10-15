import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 创建虚拟数据 (一个特征 x 和一个二元标签 y)
# 假设数据是线性可分的，以便有一个明确的“最优”区域
np.random.seed(24)
m = 20 # 样本数
X = np.linspace(-5, 5, m) # 特征 x
# 真实的标签 y (0 或 1)
y = (X > 0).astype(int)

# 2. 定义逻辑回归模型 (Sigmoid 函数)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. 定义使用 SSE 的损失函数 J(beta0, beta1)
def sse_loss(X, y, beta0, beta1):
    m = len(y)
    # 线性组合 z
    z = beta0 + beta1 * X
    # 预测概率 P
    P = sigmoid(z)
    # SSE 损失
    loss = np.sum((y - P)**2)
    return loss

# 4. 计算损失：在参数网格上
# 定义参数范围
beta0_range = np.linspace(-5, 5, 50)
beta1_range = np.linspace(-5, 5, 50)
B0, B1 = np.meshgrid(beta0_range, beta1_range)

# 计算每个参数组合对应的损失值
J = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        J[i, j] = sse_loss(X, y, B0[i, j], B1[i, j])

# 5. 绘制 3D 图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(B0, B1, J, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

# 设置标签和标题
ax.set_xlabel(r'$\beta_0$ (b)', fontsize=12)
ax.set_ylabel(r'$\beta_1$ (w)', fontsize=12)
ax.set_zlabel('SSE Loss $J(\mathbf{\\beta})$', fontsize=12)
ax.set_title('SSE Loss 3D map (Non-convex)', fontsize=14)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5, label='loss')

plt.show()
