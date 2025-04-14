import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 레벨 설정: 0=모두 출력, 3=오류만 출력

import warnings
warnings.filterwarnings('ignore')  # Python 경고 무시

import sys

# 원래 stderr 저장하고, 일시적으로 stderr를 devnull로 리디렉션
stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')

# MNIST 데이터 로드를 stderr 억제된 상태에서 실행
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 데이터 로드 후 stderr를 원래 상태로 복원
sys.stderr = stderr_original

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 이 시점부터는 print는 stdout을 통해 정상 출력되고,
# 기존에 발생하는 stderr 메시지는 이미 억제된 상태여서 보이지 않습니다.
print(tf.__version__)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


def plot_mnist(data, classes, incorrect=None):
    for i in range(10):
        idxs = (classes == i)
        if incorrect is not None:
            idxs *= incorrect
        images = data[idxs][0:10]
        for j in range(images.shape[0]):
            plt.subplot(5,10, i+j*10+1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            if j == 0:
                plt.title(i)
            plt.axis('off')
    plt.show()