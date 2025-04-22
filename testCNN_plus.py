import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #  TensorFlow 로그 레벨 설정: 0=모두 출력, 3=오류만 출력

import warnings
warnings.filterwarnings('ignore')  # Python 경고 무시

import sys

# 원래 stderr 저장하고, 일시적으로 stderr를 devnull로 리디렉션
stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')

# MNIST 데이터 로드를 stderr 억제된 상태에서 실행
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#1.x 전용
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
#데이터셋 가져오는 코드
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
#데이터셋 형태 확인

def plot_mnist(data, classes, incorrect=None):
    #이미지 출력 함수. data는 이미지, classes는 라벨. True 값만 시각화한다.
    for i in range(10):#숫자 0부터 9까지
        idxs = (classes == i) #숫자 별로
        if incorrect is not None:#incorrect 안넣으면 상관없음.
            idxs *= incorrect

        images = data[idxs][0:10] #데이터 10개를 뽑음.
        for j in range(5): #근데 5개만 돌림..?
            plt.subplot(5,10, i+j*10+1)
            #5행 10열짜리 이미지 격자. i+j*10+1번째 그림. 번호가 1부터 시작함.
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            #이미지를 화면에 출력. gray는 흑백 출력을 의미. reshape로 28*28픽셀로 변환함.
            if j == 0:
                plt.title(i) #숫자가 바뀔때마다 제목 적음.
            plt.axis('off') #이거 켜면 눈금나옴. 더러워짐.
    plt.show() #이미지 출력!
classes = np.argmax(y_train, 1)
#원 핫 인코딩 방식을 숫자로 변환함. {0,0,1,0,..} -> 2
plot_mnist(x_train, classes)
#plot 함수 호출. train 이미지를 띄워줌.


import tensorflow as tf

learning_rate = 1e-4
num_of_iter = 800
batch_size = 50
keep_prob_train = 0.5
keep_prob_test = 1.0
# 파라미터. 학습률, 학습횟수, 뱃치 정의. train 학습 시 50%만 사용.


def variable_summaries(var, name):
    #통계값 요약 함수
    with tf.name_scope('summaries'):
        #with 문법은 블록 안에서 리소스를 자동으로 열고, 자동으로 닫거나 정리해 줌. 1.x전용.
        #tensorflow에서는 연산들을 그룹으로 묶을 때 사용. 시각화 시 구분을 쉽게 해줌.
        #summaries라는 그룹 이름을 붙임.
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        #평균 계산해서 저장
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        #표준편차 저장. 계산이 복잡해서 따로 묶어준듯?
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        #최대, 최소값 저장
        tf.summary.histogram(name, var)
        #히스토그램



with tf.name_scope('inputs'):
    #input data를 담는데 placeholder 사용. 1.x에서만 사용.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    #입력 이미지.
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    #정답 레이블
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    #드롭아웃 확률.



weights={
    'W_conv1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1, name='W_conv1')),
    #합성곱 레이어의 필터. [커널y, 커널x, 입력채널, 출력채널]. 입력->히든1
    #입력 이미지 1장을 대상으로 5*5 필터 32개 만들어서 32개의 feature가 나옴.
    'W_conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, name='W_conv2')),
    #[커널y, 커널x, 저차원 feature, 고차원 feature]. 히든1 -> 히든2
    #32개의 feature map을 받아서 64개의 더 고차원 feature map 생성.
    'W_fc1': tf.Variable(tf.truncated_normal([7* 7* 64, 1024], stddev=0.1, name='W_fc1')),
    #[입력 y, 입력 x, feature map, 출력]. 히든2 -> 히든3
    #[28*28]-> pooling-> [14*14] -> pool2 -> [7*7]. 따라서 입력은 7*7픽셀임.
    #합성곱층 출력을 일렬로 펴서 완전 연결층으로 전달.
    'W_fc2': tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1, name='W_fc2')),
    #히든3->출력. 10개의 숫자 중 하나로 출력됨.
}
#가중치. dictionary를 사용하여 변수를 관리하기 편함. 이름을 통해 접근 가능.

biases ={
    'b_conv1': tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1'),
    'b_conv2': tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2'),
    'b_fc1': tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1'),
    'b_fc2': tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc2'),
}
#bias를 전부 0.1로 채움.


for key, value in weights.items():
    variable_summaries(value, key)
    #weights는 가중치 딕셔너리. key는 "W_conv1" 등 문자열 이름. key는 텐서.
    #variable_summaries 함수 호출. 텐서에 대해 각종 정보 요약으로 저장
for key, value in biases.items():
    variable_summaries(value, key)
#tensorboard 시각화를 위한 요약 데이터 자동 생성


def CNN(x, weights, biases, keep_prob):
    #핵심함수!! CNN모델이다.
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image_reshape")
        #[배치 크기(자동계산), 가로 픽셀, 세로 픽셀, 채널 수(흑백이라 1)]
        tf.summary.image('input', x_image, 10)
        #시각화를 위한 summary. 이미지에서 앞에서 10개만 시각화
        print("input x_image_shape: ", x_image.get_shape().as_list())
        #현재 텐서의 모양을 리스트로 출력. [100,28,28,1]식으로 나옴.
        #그룹화된 연산. reshape하는 부분 정의.

    with tf.name_scope("Conv_Layer1"):
        # 첫번째 합성곱 레이어(conv1)를 정의.
        conv1 = tf.nn.conv2d(x_image, weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME')
        #합성곱 연산. x_image는 입력 이미지. weights는 첫번째 층의 커널.
        #strides는 한칸씩 움직인다는 뜻. padding=SAME은 패딩을 넣어서 출력 크기를 입력 크기와 동일하게 맞춤.
        print("1번째 CNN 통과 후 shape: ", conv1.get_shape().as_list())
        h_conv1 = tf.nn.relu(conv1 + biases['b_conv1'], name="h_conv1")
        #합성곱 결과인 conv1에 bias를 더하고 relu 활성화 함수를 적용한 결과. shape는 같음.


    with tf.name_scope("Pooling_Layer1"):
        #첫번째 Pooling Layer를 정의.
        h_pool1= tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
        #핵심 코드. h_conv1은 RELU 통과 결과. [batch, height, width, channels]. padding은 안짤리게.
        # ksize는 2*2 크기의 필터 사용. strides는 풀링이 안겹치게 한번에 2칸씩 이동.
        #결과적으로 28*28 -> 14*14
        print("1번째 Max Pooling 통과 후 shape: ", h_pool1.get_shape().as_list())


    with tf.name_scope("Conv_Layer2"):
        #두번째 합성곱 레이어(conv2)
        conv2 = tf.nn.conv2d(h_pool1, weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME')
        #두번째 합성곱 연산. h_pool은 입력, 커널은 W_conv2, strides는 한칸씩, padding은 안짤리게 함.
        print("2번째 CNN 통과 후 shape: ", conv2.get_shape().as_list())
        h_conv2 = tf.nn.relu(conv2 + biases['b_conv2'], name="h_conv2")
        #합성곱 결과에 bias를 더한뒤 relu 활성화 함수 적용. 결과가 출력됨.

    h_pool2= tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    #Max Pooling 레이어 적용. ksize는 2*2. strides는 안겹치게. 기본적으로 pooling1과 동일.
    print("2번째 Max pooling 통과 후 shape: ", h_pool2.get_shape().as_list())

    with tf.name_scope("Pooling_Layer2"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
        #flattening 연산. h_pool은 [batch_size,7,7,64]형태의 4D 텐서.
        #[-1, 7*7*64], 즉 각 특성맵을 일렬로 펼친 3136 차원의 벡터로 변환.

        print("Flat할 때 shape: ", h_pool2_flat.get_shape().as_list())

    with tf.name_scope("FC_Layer1"):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['W_fc1']) + biases['b_fc1'], name="h_fc1")
        #h_pool2_flat은 [batch size, 3136]. 1D 벡터. W_fc1는[3136,1024]. bias는[1024]
        #tf.matmul은 행렬곱으로 세로운 벡터를 생성함. tf.nn.relu 활성화 함수를 사용.
        print("1번째 FC Layer 통과 후 shape: ", h_fc1.get_shape().as_list())

    with tf.name_scope('dropout'):
        #드롭아웃 연산 수행. keep_prob는 몇 프로의 뉴런을 유지하냐는것. 여기선 train기준 50%.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

    with tf.name_scope("FC_Layer2"):
        #출력층! 10개의 숫자 중 하나를 출력함. 어차피 나중에 softmax 적용이 되어 활성화 함수가 없음.
        y_conv = tf.matmul(h_fc1_drop, weights['W_fc2']) + biases['b_fc2']
        #드롭아웃 후 출력값 h_fc1_drop에 대해 행렬곱. 가중치 W_fc2는 [1024,10]
        print("2번째 FC Layer 통과 후 shape: ", y_conv.get_shape().as_list())
        tf.summary.histogram('y_conv', y_conv)
        #결과값인 y_conv를 텐서보드에서 히스토그램 형태로 추적 가능하게 함. 훈련 중 분포 변화 볼 수 있음.
    return y_conv, h_fc1
    #y_conv는 모델의 출력값. h_fc1은 FC_Layer1의 출력으로, 나중에 특성 시각화나 분석에 유용.




y_conv, h_fc1 = CNN(x, weights, biases, keep_prob)
#CNN 호출


with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_), name="x_entropy")
    #손실 함수 정의. softmax_cross_entropy_with_logits는 모델의 출력(logits)과 실제 정답(labels)을 비교해서 손실을 계산. reduce_mean은 평균을 구함.
    tf.summary.scalar('x_entropy', cross_entropy)
    #텐서보드에서 그래프로 시각화할 수 있도록 요약 정보를 생성.



with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    #원래 없었는데 추가함.



with tf.name_scope('predictions'):
    #모델의 예측을 정의.
    prediction = tf.argmax(y_conv, 1)
    #y_conv는 출력값으로, 각 클래스의 확률 분포를 가지는 벡터. tf.argmax는 가장 큰 값의 인덱스를 출력.(0~9)
with tf.name_scope('target'):
    target = tf.argmax(y_, 1)
    #y_는 실제 정답을 나타내는 원-핫-인코딩 벡터.


with tf.name_scope('correct_predictions'):
    #예측 결과와 실제 정답을 비교하여 맞은 예측과 틀린 예측을 구분하는 부분.
    correct_predictions = tf.equal(prediction, target, name="correct_prediction")
    #tf.equal은 prediction과 target이 같으면 true, 아니면 false가 됨.
with tf.name_scope('incorrect_predictions'):
    incorrect_predictions = tf.not_equal(prediction, target, name="incorrect_prediction")
    #이건 반대로 prediction과 target이 같으면 false, 아니면 true를 리턴함.
    #개인적으로 굳이 필요한지 의문. coorect_prediction만 있어도 되지 않나?

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
    #정확도 계산. tf.cast는 true를 1.0으로, false를 0.0으로 바꾸고 평균, 즉 정확도를 구함.
    tf.summary.scalar('accuracy', accuracy)
    #tensorflow의 요약 기능을 사용하여 정확도를 기록.


merged = tf.summary.merge_all()
#요약 기능을 사용하여 여러 개의 요약, 즉 summary 연산을 하나로 합침.
#이 코드에서는 각종 summary.scalar 연산의 답, 각종 histogram, 이미지 등을 사용했었음.


sess = tf.InteractiveSession()
#tensorflow 1.x 고유 코드. 기본 세션 설정. 연산을 정의하고 세션을 사용해야 하기 때문.


init = tf.global_variables_initializer()
#tensorflow 1.x 코드. 모든 변수를 한번에 초기화하는 연산.


v = sess.run(init)
#변수 초기화 연산을 실제로 실행.

train_writer = tf.summary.FileWriter("./train", sess.graph)
test_writer = tf.summary.FileWriter("./test")
#tf.summary 저장경로 설정



for i in range(num_of_iter):
    #CNN 실제로 학습. 위에서 설정한대로 800번 반복함.
    batch = mnist.train.next_batch(batch_size)
    #MNIST 데이터셋에서 batch_size만큼 샘플 무작위로 뽑아옴.
    if i%100==0: #100번마다 출력
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:keep_prob_test})
        #현재 배치에 대한 정확도 계산. accuracy는 정확도를 계산하는 텐서. . eval()은 sess.run(accuracy,...)와 동일.
        #feed_dict는 계산에 사용할 실제 데이터를 넘겨줌.
        #batch[0]은 입력 이미지들로, [batch size, 784]. batch[1]는 출력,[batch size,10]. keep_prob는 drobout 체크.
        print("step %d, training accuracy %g" % (i, train_accuracy))
    summary, _ = sess.run(
        [merged, train_step], feed_dict={x:batch[0], y_:batch[1], keep_prob: keep_prob_train})
    train_writer.add_summary(summary, i)
    #train_step은




acc, pred, incorrect_pred = sess.run([accuracy, prediction, incorrect_predictions],
        feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print("test accuracy: ",acc)
print("Incorrect prediction Case")
plot_mnist(mnist.test.images, classes=pred, incorrect=incorrect_pred)



log_dir = 'logs/'


saver = tf.train.Saver()


if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)



save_path = saver.save(sess, log_dir + 'model.ckpt')
print("Model saved in path: %s" % save_path)




from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
from tensorflow.contrib.tensorboard.plugins import projector


features = sess.run(h_fc1, feed_dict={x: x_test, y_: y_test, keep_prob: keep_prob_test })


embedding_var = tf.Variable(features, name='embedding')

tf.global_variables_initializer().run()


log_dir = 'logs/'

summary_writer = tf.summary.FileWriter(
    log_dir, graph=tf.get_default_graph())
config = projector.ProjectorConfig()

embedding = config.embeddings.add()

embedding.tensor_name = embedding_var.name


metadata_path='logs/metadata.tsv'
with open(metadata_path, 'w') as f:
    for i, label in enumerate(mnist.train.labels):
        f.write('{}          \n'.format(label))
    embedding.metadata_path = metadata_path

    embedding.sprite.image_path = './mnist/mnist_10k_sprite.png'
    embedding.sprite.single_image_dim.extend([28,28])

    projector.visualize_embeddings(summary_writer, config)