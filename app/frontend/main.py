import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

API_URL = os.getenv("API_URL", "http://localhost:8000")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

app = FastAPI(title="California Housing Frontend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dự đoán giá nhà California</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 980px; padding: 0 1.25rem; background: #f6f8fb; color: #1f2937; }}
        .card {{ background: white; border-radius: 18px; padding: 2rem; box-shadow: 0 12px 30px rgba(15,23,42,.08); }}
        h1 {{ margin-bottom: 0.5rem; }}
        .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 1rem; }}
        .field {{ display: flex; flex-direction: column; gap: 6px; }}
        label {{ font-size: 0.9rem; font-weight: 600; }}
        input, select {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 0.7rem 0.8rem; font-size: 1rem; }}
        button {{ margin-top: 1rem; background: #2563eb; color: white; border: none; border-radius: 10px; padding: 0.9rem 1.4rem; cursor: pointer; font-weight: 600; }}
        .result {{ margin-top: 1.5rem; background: #ecfdf5; border: 1px solid #86efac; border-radius: 12px; padding: 1rem; }}
        .status {{ color: #475569; margin-top: 0.75rem; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Dự đoán giá nhà California</h1>
        <p>Nhập thông tin bất động sản để ước tính giá trị trung bình của căn nhà.</p>
        <form id="prediction-form">
            <div class="two-col">
                <div class="field"><label for="longitude">Kinh độ</label><input id="longitude" name="longitude" type="number" step="0.01" placeholder="-118.24" /></div>
                <div class="field"><label for="latitude">Vĩ độ</label><input id="latitude" name="latitude" type="number" step="0.01" placeholder="34.05" /></div>
                <div class="field"><label for="housing_median_age">Tuổi trung bình nhà</label><input id="housing_median_age" name="housing_median_age" type="number" step="0.1" placeholder="30" /></div>
                <div class="field"><label for="total_rooms">Tổng phòng</label><input id="total_rooms" name="total_rooms" type="number" step="1" placeholder="2400" /></div>
                <div class="field"><label for="total_bedrooms">Tổng phòng ngủ</label><input id="total_bedrooms" name="total_bedrooms" type="number" step="1" placeholder="500" /></div>
                <div class="field"><label for="population">Dân số</label><input id="population" name="population" type="number" step="1" placeholder="1200" /></div>
                <div class="field"><label for="households">Số hộ gia đình</label><input id="households" name="households" type="number" step="1" placeholder="400" /></div>
                <div class="field"><label for="median_income">Thu nhập trung bình</label><input id="median_income" name="median_income" type="number" step="0.01" placeholder="4.5" /></div>
                <div class="field" style="grid-column: 1 / -1;"><label for="ocean_proximity">Vị trí gần biển</label>
                    <select id="ocean_proximity" name="ocean_proximity">
                        <option value="" selected disabled>Chọn một tùy chọn</option>
                        <option value="<1H OCEAN"><1H OCEAN</option>
                        <option value="INLAND">INLAND</option>
                        <option value="ISLAND">ISLAND</option>
                        <option value="NEAR BAY">NEAR BAY</option>
                        <option value="NEAR OCEAN">NEAR OCEAN</option>
                    </select>
                </div>
            </div>
            <button type="submit">Dự đoán</button>
        </form>
        <div class="status" id="status">Trạng thái: chờ</div>
        <div class="result" id="result" style="display:none;"></div>
    </div>

    <script>
        const apiBase = "{API_URL}";
        const form = document.getElementById('prediction-form');
        const resultBox = document.getElementById('result');
        const statusBox = document.getElementById('status');

        const getApiErrorMessage = (data) => {{
            if (!data) return 'Dự đoán thất bại';
            if (typeof data.detail === 'string') return data.detail;
            if (Array.isArray(data.detail)) {{
                const first = data.detail[0];
                if (first?.msg) return first.msg;
                return 'Lỗi xác thực';
            }}
            if (typeof data.detail === 'object') {{
                if (typeof data.detail.detail === 'string') return data.detail.detail;
                if (Array.isArray(data.detail.detail)) {{
                    const first = data.detail.detail[0];
                    if (first?.msg) return first.msg;
                }}
                return data.detail.error || 'Dự đoán thất bại';
            }}
            return 'Dự đoán thất bại';
        }};

        form.addEventListener('submit', async (event) => {{
            event.preventDefault();
            const payload = {{
                features: {{
                    longitude: Number(document.getElementById('longitude').value),
                    latitude: Number(document.getElementById('latitude').value),
                    housing_median_age: Number(document.getElementById('housing_median_age').value),
                    total_rooms: Number(document.getElementById('total_rooms').value),
                    total_bedrooms: Number(document.getElementById('total_bedrooms').value),
                    population: Number(document.getElementById('population').value),
                    households: Number(document.getElementById('households').value),
                    median_income: Number(document.getElementById('median_income').value),
                    ocean_proximity: document.getElementById('ocean_proximity').value,
                }}
            }};

            statusBox.textContent = 'Trạng thái: đang gửi yêu cầu...';
            try {{
                const response = await fetch(`${{apiBase}}/api/predict`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                const data = await response.json().catch(() => ({{}}));
                if (!response.ok) {{
                    throw new Error(getApiErrorMessage(data));
                }}

                const prediction = Number(data.prediction).toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
                resultBox.innerHTML = `
                    <strong>Kết quả dự đoán:</strong> ${{prediction}} USD<br>
                    <strong>Phiên bản mô hình:</strong> ${{data.model_version}}<br>
                    <strong>Mã yêu cầu:</strong> ${{data.request_id}}
                `;
                resultBox.style.display = 'block';
                statusBox.textContent = 'Trạng thái: thành công';
            }} catch (error) {{
                resultBox.innerHTML = `<strong>Lỗi:</strong> ${{error.message}}`;
                resultBox.style.display = 'block';
                statusBox.textContent = 'Trạng thái: thất bại';
            }}
        }});
    </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "service": "frontend", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=FRONTEND_PORT)
