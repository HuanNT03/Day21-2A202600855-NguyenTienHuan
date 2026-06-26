# BÁO CÁO KẾT QUẢ THỰC HÀNH LAB MLOPS
## Đề tài: Hệ thống phân loại chất lượng rượu vang (Wine Quality Classification)

---

## 1. Kết Quả Thử Nghiệm Siêu Tham Số Cục Bộ (Bước 1)

Trong Bước 1, chúng ta đã chạy thử nghiệm cục bộ với 3 bộ siêu tham số khác nhau cho thuật toán `RandomForestClassifier` trên tập dữ liệu ban đầu `train_phase1.csv` (2998 mẫu) và đánh giá trên `eval.csv` (500 mẫu).

Dưới đây là bảng so sánh hiệu năng được ghi nhận từ giao diện MLflow UI:

| Lần chạy | Siêu tham số (n_estimators, max_depth, min_samples_split) | Accuracy | F1-Score (Weighted) | Nhận xét |
|:---:|---|:---:|:---:|---|
| **Lần 1** | `n_estimators: 100`, `max_depth: 5`, `min_samples_split: 2` | 0.5880 | 0.5824 | Mô hình quá đơn giản, bị underfitting. |
| **Lần 2** | `n_estimators: 50`, `max_depth: 3`, `min_samples_split: 2` | 0.5580 | 0.5492 | Độ sâu cây quá thấp, hiệu năng kém nhất. |
| **Lần 3** | `n_estimators: 256`, `max_depth: 20`, `min_samples_split: 2` (Seed: 1083) | **0.6920** | **0.6907** | Mô hình học tốt hơn, tiệm cận ngưỡng 0.70. |

> [!TIP]
> **Bộ tham số tối ưu lựa chọn:** Bộ siêu tham số ở Lần chạy 3 (`n_estimators: 256`, `max_depth: 20`, `min_samples_split: 2`) cho hiệu năng tốt nhất trên tập kiểm thử thô và được chọn làm cấu hình chính thức trong `params.yaml`.

---

## 2. Giải Pháp Điều Chỉnh Thuật Toán Đạt Ngưỡng Accuracy >= 0.70

Do dữ liệu thô ban đầu có tính phân tách kém (Random Forest cơ bản trên 12 đặc trưng chỉ đạt tối đa khoảng 0.6920, chưa vượt được ngưỡng chất lượng đánh giá 0.70 của Eval Gate), chúng ta đã thực hiện giải pháp nâng cao:

* **Feature Engineering (Kỹ nghệ đặc trưng):** Tạo thêm 6 đặc trưng mới dựa trên tri thức miền (domain knowledge) về rượu:
  1. `sulfur_dioxide_ratio`: Tỷ lệ khí SO2 tự do trên tổng SO2.
  2. `acid_ratio`: Tỷ lệ axit bay hơi trên axit cố định.
  3. `alcohol_density`: Tỷ lệ nồng độ cồn trên mật độ rượu.
  4. `alcohol_sulphates`: Sự kết hợp giữa nồng độ cồn và sunfat.
  5. `volatile_acidity_alcohol`: Tương tác giữa axit bay hơi và cồn.
  6. `total_acidity`: Tổng lượng axit trong rượu.
* **Scikit-learn Pipeline:** Để đảm bảo tính tương thích ngược tuyệt đối với API phục vụ (`serve.py`) và Unit Test (`test_train.py`), toàn bộ quá trình biến đổi đặc trưng và huấn luyện phân loại được đóng gói trong một **Pipeline**. Hệ thống deployment chỉ cần gọi `model.predict([12 đặc trưng thô])` mà không cần thay đổi bất kỳ dòng code logic nào.

---

## 3. So Sánh Hiệu Năng Trước Và Sau Khi Bổ Sung Dữ Liệu (Bước 2 vs Bước 3)

Sau khi tích hợp Pipeline với các đặc trưng cải tiến, mô hình đạt được hiệu năng vượt bậc:

| Chỉ số | Bước 2 (2998 mẫu - train_phase1) | Bước 3 (5996 mẫu - train_phase1 + phase2) | Mức độ cải thiện |
|---|:---:|:---:|:---:|
| **Accuracy** | **0.7180** | **0.7820** | **+6.40%** |
| **F1-Score** | **0.7156** | **0.7805** | **+6.49%** |

**Nhận xét:**
* Việc bổ sung thêm 2998 mẫu từ tập `train_phase2.csv` giúp mô hình tăng mạnh độ chính xác từ `71.80%` lên `78.20%`.
* Điều này chứng minh rằng mô hình học thêm được nhiều phân phối dữ liệu mới, cải thiện khả năng tổng quát hóa trên tập `eval.csv` mà không bị overfitting.

---

## 4. Các Khó Khăn Kỹ Thuật Gặp Phải Và Giải Pháp

1. **Lỗi SQLite + MLflow Proxy Artifact (Lỗi `proxy mlflow-artifact scheme`):**
   * *Khó khăn:* Khi lưu log model cục bộ bằng MLflow sử dụng SQLite làm backend (`sqlite:///mlflow.db`), MLflow mặc định cố gắng ghi artifact qua proxy không tồn tại.
   * *Giải pháp:* Chuyển đổi run sang một Experiment mới với tên cụ thể (`mlflow.set_experiment("WineQuality")`) trước khi khởi chạy `start_run()`, giúp MLflow cấu hình đường dẫn artifact cục bộ đúng định dạng `file://`.
2. **Quản lý dữ liệu lớn bằng DVC trên Git:**
   * *Khó khăn:* Các file dữ liệu lớn (.csv) không được phép push trực tiếp lên GitHub để tránh làm nặng repository.
   * *Giải pháp:* Khởi tạo DVC remote lưu trữ trên Cloud Storage (AWS S3) và commit các file con trỏ `.csv.dvc` vào Git. Cấu hình credentials trong file `config.local` để bảo mật thông tin tài khoản.
3. **Quy trình deploy qua SSH trên GitHub Actions:**
   * *Khó khăn:* Phải kết nối bảo mật từ runner của GitHub sang VM EC2 và thực hiện restart service mà không làm lộ SSH key.
   * *Giải pháp:* Sinh cặp SSH key chuyên dụng (`mlops_deploy`), lưu private key vào GitHub Secrets (`VM_SSH_KEY`) và đưa public key vào `authorized_keys` của VM EC2.
