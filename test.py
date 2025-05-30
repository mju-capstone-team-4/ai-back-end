from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
import io
import tensorflow as tf
import numpy as np

app = FastAPI()

class_name_ko = [

    '포도_검은썩음병', # 0
    '포도_에스카병(흑색홍반병)',
    '포도_정상',

    '오렌지_황룡병(감귤그리닝병)', # 3

    '고추_세균성반점병',
    '고추_정상',

    '감자_초기역병', #6
    '감자_후기역병',
    '감자_정상',

    '호박_흰가루병', #9

    '딸기_잎마름병',
    '딸기_정상',

    '토마토_세균성반점병', #12
    '토마토_모자이크바이러스', #13
    '토마토_정상', #14

    '수박_탄저병', #15
    '수박_노균병',
    '수박_정상'
]

tomato_name = [
    '토마토_세균성반점병', #0
    '토마토_모자이크바이러스',
    '토마토_정상',
]

corn_name = [
 '옥수수_회색잎반점',
 '옥수수_녹병',
 '옥수수_정상'
]

grape_name = [
    '포도_검은썩음병', # 0
    '포도_에스카병(흑색홍반병)',
    '포도_정상'
]

orange_watermelon_name = [
    '오렌지_녹화병',
    '수박_탄저병',
    '수박_흰가루병',
    '수박_정상'
]

# 모델 로드
model_main = tf.keras.models.load_model("fifth_trained_model.keras")
tomato_model = tf.keras.models.load_model("tomato_trained_model.keras")
corn_model = tf.keras.models.load_model("corn_trained_model.keras")
grape_model = tf.keras.models.load_model("grape_trained_model.keras")
orange_watermelon_model = tf.keras.models.load_model("orange_watermelon_trained_model.keras")

# 이미지 예측 함수
def predict_image(image_url, fruit):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("이미지를 다운로드할 수 없음. URL이 올바른지 확인하세요.")

    # URL에서 받은 이미지 데이터를 `tf.keras.preprocessing.image.load_img()` 방식으로 변환
    image = Image.open(io.BytesIO(response.content))
    image.save("temp_image.jpg")  # 임시 저장 후 로드 (로컬 방식과 동일하게 처리)

    
    # ✅ 로컬 방식과 동일한 전처리 적용
    image = tf.keras.preprocessing.image.load_img("temp_image.jpg", target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch

    if(fruit == '토마토'):
        prediction = tomato_model(input_arr)
        result_index = np.argmax(prediction) + 12
        result = class_name_ko[result_index] 

    elif(fruit == '옥수수'):
        prediction = corn_model(input_arr)
        result_index = np.argmax(prediction)
        result = corn_name[result_index] 
    elif(fruit == '포도'):
        prediction = grape_model(input_arr)
        result_index = np.argmax(prediction)
        result = grape_name[result_index] 
    elif(fruit == '오렌지' or fruit =='수박'):
        prediction = grape_model(input_arr)
        result_index = np.argmax(prediction)
        result = orange_watermelon_name[result_index] 

    else:
        prediction = model_main.predict(input_arr)
        result_index = np.argmax(prediction)
        result = class_name_ko[result_index] 

    confidence = float(np.max(prediction))  # 확률값 변환 (JSON 직렬화 가능하도록)
    
    print("-" * 50)
    print("모든 확률값:", prediction)
    print("-" * 50)

    print("-" * 50)
    print("정확도:", confidence)
    print("-" * 50)

    print("-" * 50)
    print("인덱스:", result_index)
    print("-" * 50)

    print("-" * 50)
    print("인덱스:" + result)
    print("-" * 50)
    
    return result, confidence


# 요청 모델
class ImageURLRequest(BaseModel):
    image_url: str
    crop: str

# API 라우트 설정
@app.post("/predict/")
async def predict(data: ImageURLRequest):
    try:
        predicted_class, confidence = predict_image(data.image_url, data.crop)
        return {"result": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

    