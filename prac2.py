import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 데이터셋 준비하기
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
                    'dataset/train',
                    target_size=(224, 224),
                    batch_size=16,
                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    'dataset/test',
                    target_size=(224, 224),
                    batch_size=16,
                    class_mode='categorical')    

x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

# train 데이터 저장
for i in range(900) :
    img, label = next(train_generator) 
    # 리스트에 추가
    x_train_list.extend(img)
    y_train_list.extend(label)

# 테스트 데이터 저장
for i in range(200) :
    img, label = next(test_generator)
    # 리스트에 추가
    x_test_list.extend(img)
    y_test_list.extend(label)

# numpy 배열로 변경
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print('-' * 20)


# 2) 모델 구성하기
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(8, activation='softmax')
])


# 모델 컴파일하기
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 4) 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=150, 
                 batch_size=8)

# 5) 모델 학습과정 살펴보기
plt.rcParams['figure.figsize'] = (10, 6)
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train_loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train_accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6) 모델 평가하기
score = model.evaluate(x_test, y_test)
print('손실 :', score[0])
print('정확도 : %.2f%%' %(score[1]*100))
