########## 2023 Fall PAML Project ##############
######### LinearSVM + KernelSVM ################

######### Yang Rong ############################

import numpy as np
from sklearn.pipeline import make_pipeline
from keras.datasets import cifar10
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import cv2
from skimage.feature import hog
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Model
from keras.layers import Input, Dense
import argparse

parser = argparse.ArgumentParser(
    description="choose HOG + PCA or Autoencoder to extract features"
)

parser.add_argument(
    "--processing", type=str, default="hog", help="choose HOG+PCA or not"
)
parser.add_argument(
    "--SVM", type=str, default="linear", help="choose linear SVM or kernel SVM"
)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

###正则系数提供不同的取值
acc_train_svm_linear, acc_test_svm_linear, c1 = (
    [],
    [],
    [0.0001, 0.001, 0.01, 0.1, 1, 10],
)


## 提取HOG特征
def hog_process(imgs):
    hog_imgs = []
    for x in tqdm(imgs):
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255
        fd = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_imgs.append(fd)
    return np.array(hog_imgs)


### 定义一个自编码器
def AutoEncoder(traingset, testset):
    input_img = Input(shape=(3072,))
    encoded = Dense(256, activation="relu")(input_img)
    decoded = Dense(3072, activation="relu")(encoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    # 训练自编码器
    autoencoder.fit(
        traingset,
        traingset,
        epochs=5,
        batch_size=256,
        shuffle=False,
        validation_data=(testset, testset),
    )

    # 使用自编码器对数据进行降维
    x_train_encoded = encoder.predict(traingset)
    x_test_encoded = encoder.predict(testset)

    return x_train_encoded, x_test_encoded


##在不同的正则系数C条件下，进行SVM分类
def testC(tc, Trainset, Trainlabel, Testset, Testlabel, c1, chooseSVM, processing):
    for c in c1:
        start = time.perf_counter()
        svm_classifier(
            Trainset, Trainlabel, Testset, Testlabel, c, chooseSVM, processing
        )
        end = time.perf_counter()
        timecost = end - start
        tc.append(timecost)
    avgtc = np.mean(tc)
    print(f"average time cost is {avgtc}")


### SVM 分类器
def svm_classifier(Trainset, Trainlabel, Testset, Testlabel, c, chooseSVM, processing):
    clf = make_pipeline(StandardScaler())
    if chooseSVM == "linear" and processing == "hog":
        clf = make_pipeline(
            StandardScaler(),
            PCA(0.8),
            LinearSVC(dual="auto", random_state=0, tol=1e-5, C=c, max_iter=1500),
        )

    if chooseSVM == "kernel" and processing == "hog":
        clf = make_pipeline(
            StandardScaler(),
            PCA(0.8),
            SVC(C=c, max_iter=1500, kernel="rbf"),
        )

    if chooseSVM == "kernel" and processing == "autoencoder":
        clf = SVC(C=c, max_iter=1500, kernel="rbf")

    if chooseSVM == "linear" and processing == "autoencoder":
        clf = LinearSVC(
            dual="auto", random_state=0, tol=1e-5, C=c, max_iter=1500
        )  ##使用SVC会比较慢，故这里用linearSVC，而不是SVC(kernel='linear')

    # 训练集
    clf.fit(Trainset, Trainlabel)
    y_pred_svm_train = clf.predict(Trainset)
    acc_train = accuracy_score(Trainlabel, y_pred_svm_train)
    acc_train_svm_linear.append(acc_train)
    print("Train Accuracy = {0:f}".format(acc_train))

    # 验证集
    y_pred_svm_test = clf.predict(Testset)
    acc_test = accuracy_score(Testlabel, y_pred_svm_test)
    acc_test_svm_linear.append(acc_test)
    print("Test Accuracy = {0:f}".format(acc_test))

    ## 仅当 c == 10时，画出confusion matrix
    if c == 10:
        cm = confusion_matrix(Testlabel, y_pred_svm_test)
        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=[
                "airplane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        )
        disp.plot()
        plt.xticks(rotation=45, ha="right")
        # plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    yTrain = np.squeeze(y_train)
    yTest = np.squeeze(y_test)
    tc = []
    oneofSVM = args.SVM
    if args.processing == "hog":
        ## 提取图片HOG特征
        xTrain = hog_process(X_train)
        xTest = hog_process(X_test)

        xTrain = xTrain.astype(np.float32)
        xTest = xTest.astype(np.float32)

        xTrain_encoder = np.reshape(xTrain, (xTrain.shape[0], -1))
        XTest_encoder = np.reshape(xTest, (xTest.shape[0], -1))

        testC(
            tc,
            xTrain_encoder,
            yTrain,
            XTest_encoder,
            yTest,
            c1,
            args.SVM,
            args.processing,
        )

    if args.processing == "autoencoder":
        xTrain = X_train.astype(np.float32)
        xTest = X_test.astype(np.float32)

        scaler = StandardScaler()
        xTrain = scaler.fit_transform(np.reshape(xTrain, (xTrain.shape[0], -1)))
        xTest = scaler.transform(np.reshape(xTest, (xTest.shape[0], -1)))

        xTrain_encoder, xTest_encoder = AutoEncoder(xTrain, xTest)

        testC(
            tc,
            xTrain_encoder,
            yTrain,
            xTest_encoder,
            yTest,
            c1,
            args.SVM,
            args.processing,
        )

    ### 画出 训练误差和验证误差随正则系数C的变化关系
    plt.figure()
    plt.plot(c1, acc_train_svm_linear, ".-", color="red", label="Train Accuracy")
    plt.plot(c1, acc_test_svm_linear, ".-", color="orange", label="Test Accuracy")
    plt.xlabel("c")
    plt.ylabel("Accuracy")
    plt.title("Plot of accuracy vs c for training and test data")
    plt.grid()
    plt.legend(["Training Accuracy", "Test Accuracy"])
    plt.show()
