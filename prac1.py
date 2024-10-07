import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, ReduceLROnPlateau
import signal
import json

# 데이터 디렉토리 설정
base_dir = r'D:\ar_class\project\dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 이미지 전처리 및 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # 학습 데이터를 90% 학습, 10% 검증으로 나눕니다.
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로드
print("학습 데이터 로드 중...")
try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # 학습 데이터로 사용
    )
    print(f'학습 데이터 제너레이터: {train_generator}')
    print(f'학습 데이터 샘플 수: {train_generator.samples}')
except Exception as e:
    print(f"학습 데이터 로드 중 오류 발생: {e}")

print("검증 데이터 로드 중...")
try:
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # 검증 데이터로 사용
    )
    print(f'검증 데이터 제너레이터: {validation_generator}')
    print(f'검증 데이터 샘플 수: {validation_generator.samples}')
except Exception as e:
    print(f"검증 데이터 로드 중 오류 발생: {e}")

print("테스트 데이터 로드 중...")
try:
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    print(f'테스트 데이터 제너레이터: {test_generator}')
    print(f'테스트 데이터 샘플 수: {test_generator.samples}')
except Exception as e:
    print(f"테스트 데이터 로드 중 오류 발생: {e}")

# 클래스 인덱스 확인
print("클래스 인덱스(학습):", train_generator.class_indices)
print("클래스 인덱스(검증):", validation_generator.class_indices)
print("클래스 인덱스(테스트):", test_generator.class_indices)

# 데이터 시각화
def plot_images(generator):
    x, y = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i])
        plt.title(f'Label: {np.argmax(y[i])}')
        plt.axis('off')
    plt.show()

plot_images(train_generator)

# 모델 정의
print("모델 생성 중...")
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
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 모델 컴파일
print("모델 컴파일 중...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 체크포인트 콜백 설정
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.keras',
    save_weights_only=False,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_freq='epoch'  # 각 에포크마다 모델 저장
)

# 조기 종료 콜백 설정
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 학습률 감소 콜백 설정
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# 모델 학습
print("모델 학습 시작...")
try:
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1
    )
    print("모델 학습 완료")
except Exception as e:
    print(f"학습 중 오류 발생: {e}")

# 클래스 인덱스 저장
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# 모델 평가
print("모델 평가 중...")
try:
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')
except Exception as e:
    print(f"평가 중 오류 발생: {e}")

# 학습 및 검증 과정에서의 손실과 정확도를 그래프로 시각화
def plot_history(history):
    if history:
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, acc, 'b-', label='Training accuracy')
        plt.plot(epochs, val_acc, 'g-', label='Validation accuracy')
        plt.plot(epochs, loss, 'r-', label='Training loss')
        plt.plot(epochs, val_loss, 'orange', label='Validation loss')
        plt.title('Training and Validation Accuracy and Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.legend()

        plt.show()
    else:
        print("플롯할 학습 기록이 없습니다.")

plot_history(history)
