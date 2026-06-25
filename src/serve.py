from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Đọc cấu hình từ biến môi trường
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong. (Được chỉnh sửa sang S3 cho AWS)

    Ham nay duoc goi mot lan khi module duoc import.
    """
    # TODO 1: Tao storage.Client()
    # client = storage.Client()
    client = boto3.client("s3", region_name=AWS_REGION)

    # TODO 2: Lay bucket va blob tuong ung
    # bucket = client.bucket(GCS_BUCKET)
    # blob   = bucket.blob(GCS_MODEL_KEY)

    # TODO 3: Tai file model xuong may
    # blob.download_to_filename(MODEL_PATH)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    # print("Model da duoc tai xuong tu GCS.")
    print("Model da duoc tai xuong tu S3 thành công.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    # TODO 6: Kiem tra so luong dac trung.
    # Neu len(req.features) != 12, raise HTTPException(status_code=400, ...)
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features, but received {len(req.features)}."
        )

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    # pred = model.predict(...)
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    # Nhan tuong ung: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"
    # return {"prediction": ..., "label": ...}
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {
        "prediction": pred,
        "label": labels.get(pred, "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
