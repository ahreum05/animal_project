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
