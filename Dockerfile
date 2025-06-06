# 베이스 이미지 선택 (TensorFlow가 포함된 Python 이미지)
FROM tensorflow/tensorflow:latest

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일들 복사
COPY requirements.txt .
COPY test.py .
COPY ./model/fifth_trained_model.keras .
COPY ./model/corn_trained_model.keras .
COPY ./model/grape_trained_model.keras .
COPY ./model/orange_watermelon_trained_model.keras .
COPY ./model/tomato_trained_model.keras .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 서버 실행
CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8020"]