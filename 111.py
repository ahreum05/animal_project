import tensorflow as tf
import tensorflow.keras.preprocessing.image 
import ImageDataGenerator, load_img, img_to_array

# 저장된 모델 로드
model = tf.keras.models.load_model('animal_model.keras')

# 모델 요약
model.summary()

# 테스트 데이터 로드
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# 예측 함수 정의
def predict_human_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # 이미지 로드 및 리사이즈
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 이미지 정규화

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    return predicted_class, prediction[0][predicted_class_index]

# 예측 예제
human_image_path = r'path_to_human_image.jpg'  # 예측할 사람 사진 경로
predicted_class, confidence = predict_human_image(human_image_path)
print(f'얼굴상: {predicted_class} 상, 신뢰도: {confidence:.2f}')