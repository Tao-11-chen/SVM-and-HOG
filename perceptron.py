import numpy as np


# 定义激活函数
def acti_fun(x):
    return 1 if x > 0 else 0


# 创建感知器类
class Perception(object):
    # 初始化权重
    def __init__(self):
        self.weights = np.random.random()
        self.bias = 0

    # 定义训练函数，包括训练次数iter,学习率rate
    def train(self, input_vecs, labels, iter, rate):
        for i in range(iter):
            for input_vec, label in zip(input_vecs, labels):
                output = acti_fun(sum(np.array(input_vec) * self.weights) + self.bias)
                bias = label - output
                # 更新权重
                self.weights += rate * bias * np.array(input_vec)
                self.bias += rate * bias
        return self.weights, self.bias

    # 定义预测函数

    def predict(self, input_data):
        input_data = np.array(input_data)
        pred = []
        for each in input_data:
            pred_each = acti_fun(sum(np.array(each) * np.array(self.weights)) + self.bias)
            pred.append(pred_each)
        return pred


# 测试
if __name__ == '__main__':
    input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    labels = [1, 0, 0, 0]
    inputdata = [[0.5, 5], [1, 1]]  # 可以单个[1,0]，也可以多个[[-1,0],[1,1]]
    inputdata = np.mat(inputdata)
    p = Perception()
    p.train(input_vecs, labels, 1, 1)
    predict = p.predict(inputdata)
    print(predict)