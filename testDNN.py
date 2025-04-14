# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# import sys
# stderr = sys.stderr  # 원래 stderr 저장
# sys.stderr = open(os.devnull, 'w')  # stderr 무시로 바꿈
#
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# sys.stderr = stderr  # stderr 다시 원래대로 복원
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


import tensorflow as tf
print(tf.__version__)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#mnist, 즉 숫자 손글씨 데이터 import


model=tf.keras.Sequential(layers=[ #모델 정의
    tf.keras.layers.Flatten(input_shape=(28,28)),#(28,28)->(784) 1차원 벡터로
    tf.keras.layers.Dense(128, activation='relu'),#은닉층. 뉴런 128개에 relu 사용
    tf.keras.layers.Dropout(0.2),#무작위로 20%의 뉴런을 끔.
    tf.keras.layers.Dense(10, activation='softmax')#출력층. 10개의 클래스를 분류하고 각 클래스의 확률 출력.
])
model.summary()#모델의 구조를 요약해 보여줌.


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#모델을 학습시킬 준비를 함. 최적화 함수는 아담 사용.
# loss function은 정수형의 레이블에 적합한 다중 클래스 분류 손실함수. metrics는 학습 과정 중 정확도를 지표로 쓴다는 것.
history = model.fit(x_train, y_train, epochs=5)
#모델을 실제로 학습시키는 단계.시간이 다소 걸리니 주의.
# x_train은 학습할 이미지 데이터(28x28), y_train은 정답 레이블(0~9). 이걸 5번 반복.
#history에는 각 epoch마다의 loss와 정확도가 기록됨.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#모델 성능 평가(손실 및 정확도). verbose=2는 한 줄 요약 형식으로 출력함.
print('\n테스트 정확도:', test_acc)


#결과 이미지로 출력 및 예측
import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(x_test)
#예측 수행. 결과는 (10000, 10) 형태의 확률분포배열.
pred=np.argmax(predictions[0])
#예측 숫자, 즉 확률이 가장 큰 값의 인덱스 반환.
print("예측값: {}, 실제값:{}".format(pred,y_test[0]))
plt.imshow(x_test[0])
#이미지를 화면에 준비.
plt.show()
#이미지 출력

#도표로 출력
print(history.history)
plt.plot(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.legend(["loss", "accuracy"])
plt.show()
