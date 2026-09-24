# California Housing Price Prediction

## 1. Giới thiệu đề tài

Dự án này xây dựng hệ thống dự đoán giá nhà California bằng cách kết hợp quy trình học máy, API, và ứng dụng web chạy trên Docker. Mục tiêu là chuyển mô hình từ notebook Colab sang sản phẩm có thể dùng thực tế với các endpoint rõ ràng, nhật ký theo `request_id`, lưu lịch sử dự đoán và khả năng triển khai qua tunnel/ngrok hoặc Docker Compose trên máy cục bộ.

## 2. Mục tiêu và phạm vi bài tập

- Tạo mô hình hồi quy dự đoán `median_house_value` từ dataset California Housing.
- Đóng gói model dưới dạng file `model.joblib` có kèm `schema.json` và `metadata.json`.
- Thiết kế kiến trúc 3 tầng: AI Service, Backend, Frontend.
- Đưa toàn bộ hệ thống lên Docker Compose với MongoDB.
- Đảm bảo luồng dữ liệu FE → BE → AI Service → FE hoạt động đúng, có log mã request và lưu lịch sử dự đoán vào MongoDB.

## 3. Tóm tắt kiến trúc hệ thống

```mermaid
flowchart LR
    U[User] --> FE[Frontend]
    FE --> BE[Backend FastAPI]
    BE --> AI[AI Service FastAPI]
    BE --> DB[(MongoDB)]
    AI --> M[model.joblib]
```

- AI Service: nạp model khi khởi động container.
- Backend: validate đầu vào theo `schema.json`, gọi AI Service, lưu lịch sử dự đoán.
- Frontend: form nhập dữ liệu, gửi lên backend và hiển thị kết quả.

## 4. Dữ liệu và mô hình

Dataset được dùng là California Housing, gồm các đặc trưng chính như:

- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `ocean_proximity`
- `median_house_value` (biến mục tiêu)

Mô hình được lưu ở:

- `ai-models/models/model.joblib`
- `ai-models/models/schema.json`
- `ai-models/models/metadata.json`

## 5. Cấu trúc thư mục dự án

```text
.
├── .env.example
├── docker-compose.yml
├── README.md
├── ai-models/
│   ├── data/
│   ├── models/
│   ├── requirements.txt
│   ├── service/
│   └── colab/
├── app/
│   ├── backend/
│   └── frontend/
├── docs/
└── .gitignore
```

## 6. Thiết lập môi trường

Yêu cầu:

- Docker + Docker Compose
- Git
- Python 3.12 (cho môi trường phát triển cục bộ)

Tạo file biến môi trường từ mẫu:

```bash
cp .env.example .env
```

Các biến môi trường chính:

```env
AI_SERVICE_URL=http://ai-service:8001
API_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
MONGODB_URI=mongodb://mongodb:27017
MONGODB_DATABASE=cali_house_db
MONGODB_COLLECTION=predictions
```

## 7. Chạy dự án bằng Docker

Từ thư mục gốc của dự án:

```bash
docker compose up --build
```

Sau khi khởi động, các service sẽ có sẵn tại:

- AI Service: http://localhost:8001
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- MongoDB: mongodb://localhost:27017

Kiểm tra sức khỏe từng service:

```bash
curl http://localhost:8001/health
curl http://localhost:8000/health
curl http://localhost:3000/health
```

## 8. API contract

### AI Service

- `GET /health`
- `GET /model-info`
- `POST /predict`

Ví dụ request:

```json
{
  "features": {
    "longitude": -118.24,
    "latitude": 34.05,
    "housing_median_age": 30,
    "total_rooms": 2400,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 400,
    "median_income": 4.5,
    "ocean_proximity": "NEAR BAY"
  }
}
```

Ví dụ response:

```json
{
  "prediction": 432500.0,
  "model_version": "1.0.0",
  "request_id": "abc123",
  "target_column": "median_house_value"
}
```

### Backend

- `GET /health`
- `GET /model-info`
- `GET /api/history`
- `POST /api/predict`

Dữ liệu đầu vào được validate theo `schema.json` trước khi forwarded tới AI Service.

## 9. Luồng dự đoán và lưu lịch sử

Luồng hoạt động như sau:

1. Frontend hiển thị form nhập dữ liệu.
2. Frontend gửi `POST /api/predict` tới backend.
3. Backend validate dữ liệu theo `schema.json`.
4. Backend gọi AI Service `POST /predict`.
5. AI Service tính toán bằng model đã nạp sẵn khi khởi động.
6. Backend lưu record vào MongoDB và trả kết quả cho frontend. 
7. Mỗi request có `request_id` được log rõ ràng để dễ debug và bảo vệ demo.

## 10. Hướng dẫn cập nhật tunnel/ngrok

Khi đã public hệ thống qua tunnel hoặc ngrok, cần cập nhật địa chỉ mới trong `.env` và khởi động lại services nếu cần. Ví dụ:

```env
AI_SERVICE_URL=https://<your-ngrok-id>.ngrok-free.app
API_URL=https://<your-backend-domain>.ngrok-free.app
FRONTEND_URL=https://<your-frontend-domain>.ngrok-free.app
```

Lưu ý:

- Nếu ngrok đổi link sau mỗi lần khởi động, cần cập nhật lại `.env` và ghi log thời gian đổi link.
- Đối với báo cáo và demo, nên ghi rõ lần cập nhật mới nhất và link đang dùng.

## 11. Dùng thử trên trình duyệt

Mở URL:

```text
http://localhost:3000
```

Nhập các giá trị ví dụ rồi nhấn Predict. Hệ thống sẽ hiển thị giá ước lượng theo USD và log `request_id` trên console server.

## 12. Kiểm tra và bảo trì

Nên kiểm tra bằng các lệnh sau:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f ai-service
```

Nếu cần xem lịch sử dự đoán:

```bash
curl http://localhost:8000/api/history
```

## 13. Nâng cấp và triển khai tiếp theo

Có thể triển khai tiếp theo theo các cách:

- Deploy backend + AI service trên Render hoặc máy chủ Linux.
- Deploy frontend lên Vercel hoặc Nginx static hosting.
- Dùng MongoDB Atlas cho database production.
- Cập nhật tunnel/ngrok khi public ngoài internet.

## 14. Checklist bảo vệ, đánh giá, kết luận và tài liệu tham khảo

Trước khi nộp bài và bảo vệ, cần đảm bảo:

- Mô hình đã được đóng gói hoàn chỉnh trong `ai-models/models/`.
- `schema.json` khớp với dữ liệu đầu vào và pipeline model.
- AI Service nạp model ngay khi container khởi động.
- Backend validate dữ liệu và lưu lịch sử vào MongoDB.
- Frontend hiển thị dữ liệu và kết quả trực quan.
- Docker Compose chạy đúng với `docker compose up --build`.
- `README.md` có các mục rõ ràng và link public cập nhật mới nhất.
- Có demo 15 phút với 2 máy: Máy 1 để chiếu slide, Máy 2 để xem container/log và luồng request.

Dự án này không chỉ tập trung vào độ chính xác của mô hình mà còn chú trọng tới khả năng triển khai thực tế, chuẩn hóa API, kiểm tra dữ liệu, log request, và vận hành qua Docker. Đây là mức độ sẵn sàng gần với một hệ thống AI product thực thụ, đúng với yêu cầu của bài tập lớn môn Học máy cơ bản.

- California Housing dataset
- scikit-learn documentation
- FastAPI documentation
- Docker documentation
- MongoDB documentation


