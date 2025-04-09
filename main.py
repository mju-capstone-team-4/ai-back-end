from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
import io
import tensorflow as tf
import numpy as np

app = FastAPI()

# 모델 로드
model = tf.keras.models.load_model("trained_model.keras")

# 클래스 이름 리스트 (한글 번역 적용 가능)
class_name = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy',
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight',
    'Potato___healthy', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight',
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

class_name_ko = [
    '사과_겹무늬썩음병',
    '사과_검은썩음병',
    '사과_적갈색무늬병',
    '사과_정상',

    '블루베리_정상',

    '체리_흰가루병',
    '체리_정상',

    '옥수수_세르코스포라잎마름병_회색잎무늬병',
    '옥수수_일반녹병',
    '옥수수_북부잎마름병',
    '옥수수_정상',

    '포도_검은썩음병',
    '포도_에스카병(블랙미즐병)',
    '포도_잎마름병(이사리옵시스잎반점병)',
    '포도_정상',

    '오렌지_황룡병(감귤그리닝병)',

    '복숭아_세균성반점병',
    '복숭아_정상',

    '피망_세균성반점병',
    '피망_정상',

    '감자_초기역병',
    '감자_후기역병',
    '감자_정상',

    '라즈베리_정상',

    '대두_정상',

    '호박_흰가루병',

    '딸기_잎마름병',
    '딸기_정상',

    '토마토_세균성반점병',
    '토마토_초기역병',
    '토마토_후기역병',
    '토마토_잎곰팡이병',
    '토마토_점무늬잎반점병',
    '토마토_거미진드기(두점박이거미진드기)',
    '토마토_표적반점병',
    '토마토_황화잎말림바이러스',
    '토마토_모자이크바이러스',
    '토마토_정상'
]

# 이미지 예측 함수
def predict_image(image_url):
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
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction))  # 확률값 변환 (JSON 직렬화 가능하도록)
    
    print(prediction)
    print(result_index)
    # return predict_top_3_classes(input_arr)
    return class_name_ko[result_index], confidence
    

# 요청 모델
class ImageURLRequest(BaseModel):
    image_url: str

# API 라우트 설정
@app.post("/predict/")
async def predict(data: ImageURLRequest):
    try:
        predicted_class, confidence = predict_image(data.image_url)
        return {"result": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

    