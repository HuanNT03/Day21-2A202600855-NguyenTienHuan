# HƯỚNG DẪN CHI TIẾT THỰC HIỆN LAB MLOPS (AWS S3 & EC2 VERSION)
## Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục (CI/CD)

> [!NOTE]
> Bản tài liệu này được biên soạn bởi **Senior MLOps Engineer** nhằm hướng dẫn chi tiết từng bước thực hiện bài Lab MLOps (Wine Quality Classification) sử dụng hạ tầng **AWS (Amazon S3 và Amazon EC2)**, giải thích mã nguồn cần hoàn thiện ở các file TODO và cung cấp checklist chi tiết các file cùng bằng chứng cần nộp để đạt điểm tối đa (bao gồm cả phần Bonus).

---

## 1. TỔNG QUAN HỆ THỐNG & DỮ LIỆU
Hệ thống MLOps này là mô hình thực tế thu nhỏ của một hệ thống CI/CD/CT (Continuous Integration / Continuous Deployment / Continuous Training) tự động:
1. **Dữ liệu và mã nguồn** được quản lý tách biệt: Dữ liệu lớn (.csv) lưu trên Amazon S3 được quản lý thông qua **DVC con trỏ (.dvc)**; mã nguồn được quản lý bằng **Git**.
2. **Thí nghiệm** được theo dõi chặt chẽ bằng **MLflow** cục bộ (hoặc DagsHub từ xa) để ghi nhận siêu tham số (Hyperparameters) và độ đo (Metrics - Accuracy, F1-Score).
3. **Pipeline CI/CD (GitHub Actions)** tự động thực thi khi có thay đổi:
   - **Job 1 (Test):** Chạy Unit Tests trên dữ liệu giả lập nhằm phát hiện lỗi logic code sớm.
   - **Job 2 (Train):** Xác thực AWS, kéo dữ liệu gốc qua DVC, train mô hình, xuất file `metrics.json` và upload file `model.pkl` đã train lên S3 Bucket.
   - **Job 3 (Eval):** Đóng vai trò **Quality Gate** kiểm soát chất lượng mô hình, chỉ cho phép đi tiếp nếu $Accuracy \ge 0.70$.
   - **Job 4 (Deploy):** Kết nối SSH đến AWS EC2, kéo mô hình mới nhất về và khởi động lại API (FastAPI) phục vụ dự đoán thời gian thực.

---

## 2. HƯỚNG DẪN CHI TIẾT TỪNG BƯỚC THỰC HIỆN

### BƯỚC 1: Thực Nghiệm Cục Bộ & Theo Dõi Thí Nghiệm (MLflow)

#### Bước 1.1: Khởi tạo môi trường ảo và tải dữ liệu
Chạy các lệnh dưới đây tại terminal ở thư mục gốc:
```bash
# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt các thư viện trong requirements.txt
pip install -r requirements.txt

# Sinh dữ liệu Wine Quality ngẫu nhiên phân chia sẵn
python generate_data.py
```
Sau lệnh này, thư mục `data/` sẽ chứa 3 file: `train_phase1.csv` (2998 mẫu), `eval.csv` (500 mẫu), và `train_phase2.csv` (2998 mẫu).

#### Bước 1.2: Thiết lập cấu hình MLflow cục bộ
Thêm biến môi trường để MLflow ghi nhận thí nghiệm vào file database SQLite cục bộ thay vì memory:
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

#### Bước 1.3: Cấu hình siêu tham số `params.yaml`
Tạo file `params.yaml` ở thư mục gốc:
```yaml
n_estimators: 100
max_depth: 5
min_samples_split: 2
```

> [!WARNING]
> **Lưu ý đặc biệt về lỗi SQLite + MLflow Artifacts (Lỗi `proxy mlflow-artifact scheme`):**
> Khi sử dụng SQLite làm backend (`sqlite:///mlflow.db`) trên môi trường local, MLflow mặc định cố gắng log artifacts của Experiment 0 (Default Experiment) thông qua proxy `mlflow-artifacts:/` vốn chỉ hoạt động với HTTP tracking server. Để tránh lỗi này khi log model, bạn **bắt buộc** phải chuyển sang một experiment mới có tên rõ ràng (ví dụ: `WineQuality`) bằng cách gọi hàm `mlflow.set_experiment("WineQuality")` trước `mlflow.start_run()`. Khi đó, MLflow sẽ tự động cấu hình đường dẫn lưu artifact cục bộ hợp lệ (dạng `file://...` hoặc `./mlartifacts`).

#### Bước 1.4: Hoàn thành file `src/train.py`
Dưới đây là mã nguồn đầy đủ của [train.py](file:///home/huan/Develop/Github/Day21-Lab/Day21-2A202600855/src/train.py) sau khi hoàn thiện các TODO:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.
    """
    # TODO 1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Đảm bảo thư mục đầu ra tồn tại trước khi log
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Chuyển sang experiment WineQuality để tránh lỗi proxy artifact
    mlflow.set_experiment("WineQuality")
    with mlflow.start_run():
        # TODO 3: Ghi nhận các siêu tham số
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và huấn luyện RandomForestClassifier
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Dự đoán trên tập đánh giá và tính chỉ số
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả ra màn hình
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Lưu metrics ra file outputs/metrics.json
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 9: Lưu mô hình ra file models/model.pkl
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Trả về acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
```

#### Bước 1.5: Thực nghiệm ít nhất 3 lần để so sánh
Thay đổi giá trị trong `params.yaml` và chạy lệnh train:
1. **Lần chạy 1:** Giữ nguyên mặc định (`n_estimators: 100, max_depth: 5`). Chạy `python src/train.py`.
2. **Lần chạy 2:** Sửa `params.yaml` thành `n_estimators: 50, max_depth: 3`. Chạy `python src/train.py`.
3. **Lần chạy 3:** Sửa `params.yaml` thành `n_estimators: 200, max_depth: 10`. Chạy `python src/train.py`.

#### Bước 1.6: Xem kết quả trên MLflow UI
Khởi động MLflow server cục bộ:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Mở trình duyệt truy cập: `http://localhost:5000`. Chọn experiment **WineQuality** ở góc trái. Chọn cả 3 run và bấm **Compare** để lấy bằng chứng chụp ảnh màn hình so sánh metrics. Cập nhật tham số tốt nhất vào `params.yaml`.

---

### BƯỚC 2: Cài Đặt Hạ Tầng AWS S3, DVC & Thiết Lập Pipeline CI/CD

#### Bước 2.1: Tạo Amazon S3 Bucket
Sử dụng AWS CLI hoặc Console để tạo bucket mới. Thay thế `<BUCKET_NAME>` bằng tên bucket độc nhất:
```bash
# Tạo S3 Bucket (ví dụ khu vực ap-southeast-1)
aws s3 mb s3://<BUCKET_NAME> --region ap-southeast-1
```

#### Bước 2.2: Tạo AWS Credentials với IAM
1. Tạo một IAM User chuyên dụng (ví dụ: `mlops-lab-user`) trên AWS Console.
2. Cấp quyền truy cập S3 cho User này thông qua chính sách (Policy). Bạn có thể cấp quyền `AmazonS3FullAccess` cho đơn giản trong lab, hoặc tạo Inline Policy giới hạn quyền đọc/ghi trên riêng bucket vừa tạo (Nguyên tắc quyền tối thiểu):
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": "s3:*",
               "Resource": [
                   "arn:aws:s3:::<BUCKET_NAME>",
                   "arn:aws:s3:::<BUCKET_NAME>/*"
               ]
           }
       ]
   }
   ```
3. Tạo **Access Key** cho User này và lưu lại thông tin:
   * **Access Key ID** (ví dụ: `AKIA...`)
   * **Secret Access Key** (ví dụ: `abc...`)

#### Bước 2.3: Cài đặt DVC và Đẩy Dữ Liệu Lên AWS S3
```bash
# Khởi tạo DVC trong repo
dvc init

# Thêm remote và trỏ đến S3 Bucket của bạn
dvc remote add -d myremote s3://<BUCKET_NAME>/dvc

# Lưu cấu hình xác thực DVC cục bộ (Sử dụng flag --local để ghi vào config.local, tránh commit key lên Git)
dvc remote modify myremote --local access_key_id <AWS_ACCESS_KEY_ID>
dvc remote modify myremote --local secret_access_key <AWS_SECRET_ACCESS_KEY>

# Theo dõi các file CSV gốc
dvc add data/train_phase1.csv
dvc add data/eval.csv
dvc add data/train_phase2.csv

# Commit file con trỏ .dvc vào Git (Lưu ý: Không add file .csv gốc)
git add data/train_phase1.csv.dvc data/eval.csv.dvc data/train_phase2.csv.dvc .dvc/config .gitignore
git commit -m "feat: track datasets with DVC"

# Đẩy dữ liệu thực tế lên Amazon S3
dvc push
```
Xác nhận trên AWS S3 Console rằng các file dữ liệu đã xuất hiện dưới thư mục `dvc/` trong bucket.

#### Bước 2.4: Tạo VM Amazon EC2 Phục Vụ Model Serving
1. Tạo một instance EC2 mới:
   * **AMI:** Ubuntu 22.04 LTS (Hỗ trợ tốt x86_64).
   * **Instance Type:** `t2.micro` hoặc `t3.micro` (Được miễn phí trong Free Tier).
   * **Key Pair:** Tạo key pair mới và tải về file `.pem` (ví dụ `my-key.pem`).
2. Cấu hình **Security Group**:
   * Cho phép SSH (Port 22) từ IP của bạn.
   * Cho phép HTTP (Port 8000) từ mọi nguồn (`0.0.0.0/0`) để truy cập API FastAPI.
3. Khởi chạy instance và ghi lại địa chỉ **Public IP** của VM.

#### Bước 2.5: Cài Đặt VM Thủ Công (Một lần duy nhất)
Sử dụng file `.pem` để SSH vào instance EC2:
```bash
ssh -i /path/to/my-key.pem ubuntu@<EC2_PUBLIC_IP>
```
Bên trong EC2, chạy các lệnh cấu hình môi trường:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
# Cài đặt thư viện Python phục vụ API
pip3 install fastapi uvicorn scikit-learn joblib boto3

# Tạo các thư mục chứa code và model
mkdir -p ~/models ~/src
exit
```

#### Bước 2.6: Hoàn thành file API `src/serve.py` (Phiên bản AWS S3)
Dưới đây là mã nguồn đầy đủ của [serve.py](file:///home/huan/Develop/Github/Day21-Lab/Day21-2A202600855/src/serve.py) sau khi hoàn thiện các TODO sử dụng thư viện `boto3` để tương tác với S3:

```python
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
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")


def download_model():
    """
    Tải file model.pkl từ S3 về máy khi server khởi động.
    """
    # TODO 1: Tạo S3 Client
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    # TODO 2-3: Tải file model xuống máy
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        s3_client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        # TODO 4: In thông báo thành công
        print("Model đã được tải xuống từ AWS S3 thành công.")
    except Exception as e:
        print(f"Lỗi khi tải model từ S3: {e}")
        raise e


# Tải model từ S3 trước khi load bằng joblib
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    """
    # TODO 5: Trả về dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận chính, phân loại chất lượng rượu.
    """
    # TODO 6: Kiểm tra số lượng đặc trưng đầu vào
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features, but received {len(req.features)}."
        )

    # TODO 7: Gọi model.predict để lấy kết quả dự đoán
    # Mô hình mong muốn đầu vào dạng 2D array
    pred = int(model.predict([req.features])[0])

    # TODO 8: Trả về nhãn tương ứng
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {
        "prediction": pred,
        "label": labels.get(pred, "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
Đưa file `serve.py` lên EC2 bằng lệnh scp:
```bash
scp -i /path/to/my-key.pem src/serve.py ubuntu@<EC2_PUBLIC_IP>:~/src/serve.py
```

#### Bước 2.7: Cấu Hình Systemd Service Trên VM
SSH lại vào EC2 và tạo systemd service để quản lý và tự động restart API:
```bash
ssh -i /path/to/my-key.pem ubuntu@<EC2_PUBLIC_IP>
```
Tạo service (Hãy thay thế các thông tin Access Key, Secret Key, và Bucket của bạn vào đoạn cấu hình):
```bash
sudo tee /etc/systemd/system/mlops-serve.service > /dev/null <<EOF
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="S3_BUCKET=<YOUR_BUCKET_NAME>"
Environment="AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>"
Environment="AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>"
Environment="AWS_DEFAULT_REGION=ap-southeast-1"
ExecStart=/usr/bin/python3 /home/ubuntu/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Load daemon và enable service
sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
exit
```

#### Bước 2.8: Tạo SSH Key để GitHub Actions kết nối VM
Chạy trên máy cục bộ để sinh SSH Key deploy chuyên dụng:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"

# Thêm public key vào authorized_keys trên EC2 VM
ssh -i /path/to/my-key.pem ubuntu@<EC2_PUBLIC_IP> "echo '$(cat ~/.ssh/mlops_deploy.pub)' >> ~/.ssh/authorized_keys"
```

#### Bước 2.9: Cấu Hợp GitHub Secrets
Vào repository GitHub của bạn: **Settings > Secrets and variables > Actions > New repository secret** và cấu hình 5 secrets sau:
1. `CLOUD_CREDENTIALS`: Nội dung chuỗi JSON chứa AWS Access Key (giúp giữ nguyên cấu hình bài lab):
   ```json
   {
     "aws_access_key_id": "AKIA...",
     "aws_secret_access_key": "abc..."
   }
   ```
2. `CLOUD_BUCKET`: Tên S3 Bucket đã tạo.
3. `VM_HOST`: Địa chỉ IP Public của EC2.
4. `VM_USER`: Tên user của VM (với Ubuntu EC2 mặc định là `ubuntu`).
5. `VM_SSH_KEY`: Copy toàn bộ nội dung file private key `~/.ssh/mlops_deploy` cục bộ (bao gồm cả dòng mở đầu và kết thúc dạng `-----BEGIN OPENSSH PRIVATE KEY-----`).

#### Bước 2.10: Hoàn thành file `tests/test_train.py`
Dưới đây là mã nguồn đầy đủ của [test_train.py](file:///home/huan/Develop/Github/Day21-Lab/Day21-2A202600855/tests/test_train.py):

```python
import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để sử dụng trong test.
    """
    rng = np.random.default_rng(0)
    n = 200

    # TODO 1: Tạo mảng X có kích thước (n, len(FEATURE_NAMES)) với giá trị [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))

    # TODO 2: Tạo mảng y gồm n phần tử nguyên ngẫu nhiên trong [0, 3)
    y = rng.integers(0, 3, size=n)

    # TODO 3: Xây dựng DataFrame, thêm cột "target"
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # TODO 4: Lưu 160 dòng đầu làm tập huấn luyện, 40 dòng cuối làm tập đánh giá
    train_path = str(tmp_path / "train.csv")
    eval_path  = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path,  index=False)

    # TODO 5: Trả về (train_path, eval_path)
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực nằm trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # TODO 6: Gọi hàm train() với siêu tham số nhỏ (n_estimators=10, max_depth=3)
    acc = train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)

    # TODO 7: Kiểm tra kết quả
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 8: Kiểm tra file tồn tại và nội dung đúng định dạng
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.pkl được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    # TODO 9: Kiểm tra file model tồn tại
    assert os.path.exists("models/model.pkl")
```
*Chạy test cục bộ trước khi push code:*
```bash
pytest tests/ -v
```

#### Bước 2.11: Hoàn thiện file `.github/workflows/mlops.yml` (Phiên bản AWS S3)
Dưới đây là file cấu hình GitHub Actions hoàn chỉnh cho [mlops.yml](file:///home/huan/Develop/Github/Day21-Lab/Day21-2A202600855/.github/workflows/mlops.yml) được tối ưu cho môi trường AWS:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'data/**.dvc'
      - 'src/**.py'
      - 'params.yaml'
  workflow_dispatch:

jobs:

  # ---------------------------------------------------------------------------
  # JOB 1 - UNIT TEST
  # ---------------------------------------------------------------------------
  test:
    name: Unit Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        # TODO 1: Viết lệnh để chạy pytest trên thư mục tests/ với cờ -v
        run: pytest tests/ -v


  # ---------------------------------------------------------------------------
  # JOB 2 - TRAIN
  # ---------------------------------------------------------------------------
  train:
    name: Train
    needs: test
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.read_metrics.outputs.accuracy }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Authenticate to Cloud Storage (AWS)
        # TODO 2: Phân tích JSON credentials và export các biến môi trường cho AWS CLI/SDK
        run: |
          # Phân tích JSON từ secret và export sang GITHUB_ENV
          python -c "
          import json, os
          creds = json.loads('''${{ secrets.CLOUD_CREDENTIALS }}''')
          with open(os.environ['GITHUB_ENV'], 'a') as f:
              f.write(f'AWS_ACCESS_KEY_ID={creds[\"aws_access_key_id\"]}\n')
              f.write(f'AWS_SECRET_ACCESS_KEY={creds[\"aws_secret_access_key\"]}\n')
              f.write('AWS_DEFAULT_REGION=ap-southeast-1\n')
          "

      - name: Pull data with DVC
        # TODO 3: Pull các file dữ liệu cần thiết từ S3 Bucket
        run: dvc pull data/train_phase1.csv.dvc data/eval.csv.dvc

      - name: Train model
        run: python src/train.py

      - name: Read metrics
        id: read_metrics
        # TODO 4: Đọc giá trị "accuracy" từ outputs/metrics.json bằng Python inline.
        run: |
          ACC=$(python -c "import json; d=json.load(open('outputs/metrics.json')); print(d['accuracy'])")
          echo "accuracy=$ACC" >> $GITHUB_OUTPUT

      - name: Upload model to Cloud Storage (Amazon S3)
        # TODO 5: Sử dụng boto3 SDK để upload models/model.pkl lên S3
        env:
          CLOUD_BUCKET: ${{ secrets.CLOUD_BUCKET }}
        run: |
          python - <<'EOF'
          import os
          import boto3
          
          bucket_name = os.environ.get("CLOUD_BUCKET")
          s3_client = boto3.client("s3")
          
          s3_client.upload_file(
              "models/model.pkl",
              bucket_name,
              "models/latest/model.pkl"
          )
          print("Model uploaded to S3 successfully.")
          EOF

      - name: Save metrics as artifact
        uses: actions/upload-artifact@v4
        with:
          name: metrics
          path: outputs/metrics.json


  # ---------------------------------------------------------------------------
  # JOB 3 - EVAL
  # ---------------------------------------------------------------------------
  eval:
    name: Eval
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Check eval gate
        # TODO 6: Đọc accuracy từ output của job train. Ngưỡng >= 0.70.
        run: |
          python - <<'EOF'
          acc = float("${{ needs.train.outputs.accuracy }}")
          print(f"Evaluation Gate: Model accuracy is {acc:.4f}")
          if acc < 0.70:
              raise SystemExit(f"FAILED: Accuracy {acc:.4f} is under threshold 0.70. Deployment cancelled.")
          print("PASSED: Model quality matches criteria. Deploying...")
          EOF


  # ---------------------------------------------------------------------------
  # JOB 4 - DEPLOY
  # ---------------------------------------------------------------------------
  deploy:
    name: Deploy
    needs: eval
    runs-on: ubuntu-latest
    steps:
      - name: SSH deploy to AWS EC2
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.VM_HOST }}
          username: ${{ secrets.VM_USER }}
          key: ${{ secrets.VM_SSH_KEY }}
          script: |
            # TODO 7: Restart service mlops-serve trên EC2.
            # TODO 8: Chờ server sẵn sàng rồi gọi curl /health để xác nhận.
            sudo systemctl restart mlops-serve
            sleep 5
            curl -sf http://localhost:8000/health && echo "Health check passed." || exit 1
```

#### Bước 2.12: Tạo các file rỗng và kích hoạt Pipeline
```bash
touch src/__init__.py tests/__init__.py
git add .
git commit -m "feat: complete initial pipeline implementation with AWS S3"
git push origin main
```
Sau đó, vào tab Actions trên GitHub xem 4 job chạy thành công.

#### Bước 2.13: Bật service và Kiểm thử API trên EC2 VM
Bật service lần đầu trên EC2:
```bash
ssh -i /path/to/my-key.pem ubuntu@<EC2_PUBLIC_IP> "sudo systemctl start mlops-serve"
```
Kiểm tra endpoint API từ máy cục bộ:
```bash
export VM_IP=<YOUR_EC2_PUBLIC_IP>

# 1. Health check
curl http://$VM_IP:8000/health

# 2. Dự đoán
curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```
Kết quả mong muốn: `{"prediction": 0, "label": "thap"}`.

---

### BƯỚC 3: Mô Phỏng Bổ Sung Dữ Liệu & Huấn Luyện Liên Tục (CT)

#### Bước 3.1: Chạy script ghép dữ liệu mới
Ghép tập `train_phase2.csv` vào `train_phase1.csv` hiện tại:
```bash
python add_new_data.py
# Đầu ra mong muốn: "Cập nhật dữ liệu: 2998 -> 5996 mẫu"

# Kiểm tra số dòng thực tế
wc -l data/train_phase1.csv
# Phải hiển thị 5997 dòng (gồm cả dòng tiêu đề)
```

#### Bước 3.2: Thực hiện Cập Nhật DVC & Kích Hoạt CT Pipeline
```bash
# 1. Báo cho DVC biết file train_phase1.csv đã được sửa đổi
dvc add data/train_phase1.csv

# 2. Git commit file pointer .dvc mới
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"

# 3. ĐẨY DỮ LIỆU LÊN AWS S3 TRƯỚC
dvc push

# 4. Push code Git lên GitHub để kích hoạt pipeline tự động
git push origin main
```
Theo dõi pipeline tự động kích hoạt bởi commit dữ liệu trên GitHub Actions.

---

## 3. DANH SÁCH FILE VÀ BẰNG CHỨNG CẦN NỘP

### 3.1. Đường dẫn Repository (Github URL)
* Repo phải để chế độ **Public**.
* Cấu trúc thư mục của repository phải sạch sẽ và đúng theo yêu cầu (không commit dữ liệu thực `.csv`, `.pkl`, key bảo mật).

### 3.2. Chuỗi Ảnh Chụp Màn Hình Minh Chứng (Screenshots)
Bạn cần chuẩn bị các ảnh chụp màn hình sau:
1. **Ảnh 1 - MLflow UI:** 3 lần chạy thử nghiệm cục bộ với các siêu tham số khác nhau, ghi nhận đủ `accuracy` và `f1_score`.
2. **Ảnh 2 - GitHub Actions (Bước 2):** Cả 4 Job (**Unit Test**, **Train**, **Eval**, **Deploy**) đều có tích xanh ở lần chạy đầu tiên.
3. **Ảnh 3 - AWS S3 Console:** Hiển thị các file DVC trong thư mục `dvc/` và file model tại `models/latest/model.pkl`.
4. **Ảnh 4 - Kết Quả Gọi API:** Đầu ra terminal khi chạy `curl http://VM_IP:8000/health` và `curl -X POST http://VM_IP:8000/predict ...`.
5. **Ảnh 5 - GitHub Actions (Bước 3 - CT):** Pipeline tự động kích hoạt thành công bởi commit dữ liệu mới.

### 3.3. File Báo Cáo Ngắn (Tối đa 1 trang A4)
Nội dung báo cáo:
1. Bảng so sánh siêu tham số đã thí nghiệm (Bước 1).
2. Bảng so sánh hiệu năng của mô hình trước và sau khi thêm dữ liệu (2998 mẫu vs 5996 mẫu).
3. Các khó khăn gặp phải và giải pháp.

---

## 4. HƯỚNG DẪN THỰC HIỆN CÁC THÁCH THỨC NÂNG CAO (BONUS - AWS VERSION)

### Bonus 1: Tracking MLflow Từ Xa Với DagsHub (+4 điểm)
* **Cách làm:**
  1. Đăng ký DagsHub và kết nối repository.
  2. Thêm 3 secrets vào GitHub Secrets: `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, và `DAGSHUB_TRACKING_URI`.
  3. Cập nhật job `train` trong `.github/workflows/mlops.yml` để export các biến môi trường:
     ```yaml
     - name: Run training with DagsHub MLflow
       env:
         MLFLOW_TRACKING_URI: ${{ secrets.DAGSHUB_TRACKING_URI }}
         MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
         MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
       run: python src/train.py
     ```

### Bonus 2: Thí Nghiệm Với Nhiều Thuật Toán (+4 điểm)
* **Cách làm:**
  1. Thêm tham số `model_type` vào `params.yaml` (ví dụ: `gradient_boosting`).
  2. Cập nhật `src/train.py` để khởi tạo mô hình động:
     ```python
     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
     from sklearn.linear_model import LogisticRegression

     model_type = params.get("model_type", "random_forest")
     model_params = {k: v for k, v in params.items() if k != "model_type"}
     
     if model_type == "random_forest":
         model = RandomForestClassifier(**model_params, random_state=42)
     elif model_type == "gradient_boosting":
         model = GradientBoostingClassifier(**model_params, random_state=42)
     elif model_type == "logistic_regression":
         model = LogisticRegression(**model_params, random_state=42)
     else:
         raise ValueError(f"Unsupported model type: {model_type}")
     
     model.fit(X_train, y_train)
     ```

### Bonus 3: Báo Cáo Hiệu Suất Tự Động (+4 điểm)
* **Cách làm:**
  1. Cập nhật `src/train.py` để tính toán `classification_report` và `confusion_matrix` rồi ghi vào `outputs/report.txt`:
     ```python
     from sklearn.metrics import classification_report, confusion_matrix
     
     report_str = classification_report(y_eval, preds, target_names=["thấp", "trung bình", "cao"])
     matrix_str = str(confusion_matrix(y_eval, preds))
     
     os.makedirs("outputs", exist_ok=True)
     with open("outputs/report.txt", "w") as f:
         f.write("=== HỆ THỐNG ĐÁNH GIÁ MÔ HÌNH TỰ ĐỘNG ===\n")
         f.write(f"Accuracy: {acc:.4f}\nF1-score: {f1:.4f}\n\n")
         f.write("Confusion Matrix:\n" + matrix_str + "\n\n")
         f.write("Classification Report:\n" + report_str)
     ```
  2. Lưu thư mục `outputs/` làm artifact trong `mlops.yml` bằng `actions/upload-artifact@v4`.

### Bonus 4: Hoàn Trả Về Phiên Bản Trước (Rollback Safe Guard) (+4 điểm)
* **Cách làm:**
  1. Cập nhật job `train` trong `mlops.yml` để tải file `metrics.json` cũ từ S3 về trước khi train:
     ```yaml
     - name: Get current model performance from S3
       env:
         CLOUD_BUCKET: ${{ secrets.CLOUD_BUCKET }}
       run: |
         python - <<'EOF'
         import os
         import json
         import boto3
         
         bucket_name = os.environ.get("CLOUD_BUCKET")
         s3_client = boto3.client("s3")
         
         os.makedirs("outputs", exist_ok=True)
         try:
             s3_client.download_file(bucket_name, "models/latest/metrics.json", "outputs/old_metrics.json")
             print("Tải metrics cũ từ S3 thành công.")
         except Exception:
             # Nếu chưa có model cũ thì ghi accuracy = 0
             with open("outputs/old_metrics.json", "w") as f:
                 json.dump({"accuracy": 0.0}, f)
         EOF
     ```
  2. Cập nhật Job `eval` trong `mlops.yml` để so sánh và chặn deploy:
     ```yaml
     - name: Compare accuracy with previous version
       run: |
         python - <<'EOF'
         import json
         
         with open("outputs/metrics.json") as f:
             new_acc = json.load(f)["accuracy"]
         with open("outputs/old_metrics.json") as f:
             old_acc = json.load(f).get("accuracy", 0.0)
             
         print(f"Old accuracy: {old_acc:.4f} | New accuracy: {new_acc:.4f}")
         if new_acc < old_acc:
             raise SystemExit(f"FAILED: Mô hình mới tệ hơn mô hình cũ ({new_acc:.4f} < {old_acc:.4f}). Hủy deploy.")
         print("PASSED: Đạt điều kiện cải tiến.")
         EOF
     ```
  3. Đừng quên upload `metrics.json` mới lên bucket `models/latest/metrics.json` trong job `train` để lưu lịch sử cho lần so sánh sau.

### Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu (Data Drift / Imbalance Warning) (+4 điểm)
* **Cách làm:**
  1. Thêm hàm kiểm tra phân phối dữ liệu trong `src/train.py`:
     ```python
     class_counts = df_train["target"].value_counts(normalize=True).to_dict()
     print("Label distribution:")
     for cls, pct in class_counts.items():
         print(f"Class {cls}: {pct*100:.2f}%")
         if pct < 0.10:
             print(f"WARNING: Class {cls} có tỷ lệ dưới 10%. Dữ liệu huấn luyện đang bị mất cân bằng!")
     ```
  2. Lưu thông tin phân phối nhãn này vào file `outputs/metrics.json`.

---
*Chúc các bạn hoàn thành bài Lab xuất sắc và đạt điểm số tối đa cùng với AWS!*
