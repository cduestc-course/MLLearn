import numpy as np

def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数
    :param num_examples: 数据集样本数
    :param w: 线性方程系数
    :param bias: 是否包含截距
    :param delta: 扰动项大小
    :param deg: 多项式的最高次数
    :return: features, labels
    例子：
    features, labels = arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1)
    生成一个包含1000个样本，3个特征（2个线性特征和1个截距）的线性回归数据集
    features, labels = arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 2, deg = 2)
    生成一个包含1000个样本，3个特征（2个二次特征和1个截距）的二次回归数据集

    默认创建一个 y = 2x_1 - x_2 + 1 的数据集
    """
    if bias == False:
        num_inputs = len(w)
        features = np.random.randn(num_examples, num_inputs)
        w_true = np.array(w).reshape(-1, 1)
        labels_true = np.power(features, deg).dot(w_true)
        print(features[:10])
    else:
        num_inputs = len(w) - 1
        features_true = np.random.randn(num_examples, num_inputs)
        w_true = np.array(w[:-1]).reshape(-1, 1)
        b_true = np.array(w[-1])
        labels_true = np.power(features_true, deg).dot(w_true) + b_true
        features = np.concatenate((features_true, np.ones_like(labels_true)), axis=1)
        # print(features_true[:10])
        # print(features[:10])
    labels = labels_true + np.random.normal(size = labels_true.shape) * delta

    return features, labels

def SSELoss(X, w, y):
    """SSE计算函数
    :param X: 输入数据的特征矩阵
    :param w: 线性方程参数
    :param y: 输入数据的标签数组
    :return SSE: 返回对应数据集预测结果和真实结果的误差平方和 
    """
    y_hat = X.dot(w)
    SSE = (y - y_hat).T.dot(y - y_hat)
    return SSE

def array_split(features, labels, rate=0.7, random_state=24):
    """训练集和测试集切分函数
    :param features: 特征数据
    :param labels: 标签数据
    :param rate: 训练集比例
    :param random_state: 随机种子
    :return: Xtrain, Xtest, ytrain, ytest: 训练集特征，测试集特征，训练集标签，测试集标签
    """

    arr = np.concatenate((features, labels), axis=1)
    np.random.seed(random_state)
    np.random.shuffle(arr)

    num_input = len(labels)               # 总数据量
    split_indices = int(num_input * rate) # 数据集划分的标记指标
    train_arr, test_arr = np.vsplit(arr, [split_indices, ])
    Xtrain, Xtest = train_arr[:, :-1], test_arr[:, :-1]
    ytrain, ytest = train_arr[:, -1].reshape(-1, 1), test_arr[:, -1].reshape(-1, 1)
    return Xtrain, Xtest, ytrain, ytest