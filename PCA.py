import numpy as np
from sklearn.decomposition import PCA
# 将二维数据降成1维
# num = [-3/4,-1/4,-1/4],[1/4,-1/4,-1/4],[1/4,-1/4,3/4],[1/4,3/4,-1/4]
# num_array = np.array([[0,0,0],[1,0,0],[1,0,1],[1,1,0]])
num_array = np.array([[0,0,0],[1,0,0],[1,0,1],[1,1,0],[0,0,1],[0,1,0],[0,1,1],[1,1,1]])
print(num_array)
n1_avg, n2_avg, n3_avg = np.mean(num_array[:, 0]), np.mean(num_array[:, 1]), np.mean(num_array[:, 2])
# n1_avg, n2_avg, n3_avg = np.mean(num_array[:, 0]), np.mean(num_array[:, 1]), np.mean(num_array[:, 2])
# 1.样本中心化
# new_num_array = np.array(list(zip(num_array[:, 0] - n1_avg, num_array[:, 1] - n2_avg)))
new_num_array = np.c_[num_array[:, 0] - n1_avg, num_array[:, 1] - n2_avg, num_array[:, 2] - n3_avg]
print(new_num_array)
# 2.计算协方差矩阵
# num_cov1 = np.cov(new_num_array[:, 0], new_num_array[:, 1])
# ss = new_num_array[:,0:2]
# print(ss)
# num_cov2 = np.cov(ss.T)
# print(num_cov1)
# print(num_cov2)
num_cov = np.cov(new_num_array.T)
# print(num_cov)
# 3.特征值分解
# a 特征值, b 特征向量
a, b = np.linalg.eig(num_cov)
# k=1, 取a最大值的索引对应b的特征向量
print(a)
print(b)
# w = b[:, np.argmax(a)]
# w = np.array([[0.81649658,0.40824829, 0.40824829],[0.22009329,-0.57088596,0.79097925]])
w = np.array([[1,0,0],[0,1,0]])
# print(w)
# 4.输出pca降维结果
# z1_num = new_num_array.dot(w.T)
z1_num = np.dot(w,new_num_array.T)
# print(z1_num)
reconstruct_x1 = np.dot(w.T,z1_num)
# print(reconstruct_x1)

# 使用sklearn中的PCA
pca = PCA(n_components=2)
z2_num = pca.fit_transform(num_array)
# print(z2_num)