# 🏠 House Price Prediction 

Một project dự đoán giá nhà sử dụng Machine Learning được xây dựng với Python.


## 🎯 Giới thiệu

Project này sử dụng bộ dữ liệu California Housing để dự đoán giá nhà dựa trên các đặc trưng như vị trí địa lý, thu nhập trung bình, số phòng, v.v. Được xây dựng theo kiến trúc OOP để dễ dàng bảo trì, mở rộng và tái sử dụng.

### 🔍 Bộ dữ liệu
- **Nguồn**: California Housing Dataset
- **Số lượng mẫu**: ~20,000 records
- **Đặc trưng**: 8 đặc trưng gốc + các đặc trưng được tạo thêm
- **Mục tiêu**: Dự đoán `median_house_value`

## ✨ Tính năng

- 🔄 **Pipeline tự động**: Xử lý dữ liệu từ A-Z
- 📊 **Phân tích dữ liệu**: Phân tích phân phối, missing values, outliers
- 🛠️ **Feature Engineering**: Tạo đặc trưng mới và polynomial features
- 🧹 **Data Cleaning**: Xử lý missing values, outliers, độ lệch phân phối
- 🤖 **Machine Learning**: Linear Regression với khả năng mở rộng
- 📈 **Đánh giá mô hình**: MAE, MSE, R², MAPE
- 🏗️ **Kiến trúc OOP**: Dễ bảo trì và mở rộng

## 🏛️ Kiến trúc hệ thống

```
HousePricePredictionPipeline
├── DataLoader          # Tải dữ liệu
├── DataAnalyzer        # Phân tích dữ liệu
├── DataPreprocessor    # Xử lý dữ liệu
├── ModelTrainer        # Huấn luyện mô hình
└── ModelEvaluator      # Đánh giá mô hình
```

### 📦 Các lớp chính

| Lớp | Chức năng |
|-----|-----------|
| `DataLoader` | Tải dữ liệu từ URL và hiển thị thông tin cơ bản |
| `DataAnalyzer` | Phân tích missing values, phân phối dữ liệu |
| `DataPreprocessor` | Xử lý dữ liệu, tạo features, xử lý outliers |
| `ModelTrainer` | Chia dữ liệu, huấn luyện mô hình |
| `ModelEvaluator` | Đánh giá hiệu suất mô hình |
| `HousePricePredictionPipeline` | Điều phối toàn bộ quy trình |

### Cài đặt thư viện

```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib
```

Hoặc sử dụng requirements.txt:

```bash
pip install -r requirements.txt
```

## 💻 Sử dụng

### Chạy pipeline hoàn chỉnh

```python
from house_price_prediction import HousePricePredictionPipeline

# Khởi tạo và chạy pipeline
pipeline = HousePricePredictionPipeline()
results = pipeline.run_pipeline()

print(f"R² Score: {results['r2']:.4f}")
```

### Sử dụng từng component riêng lẻ

```python
# Tải dữ liệu
loader = DataLoader()
data = loader.load_data()

# Phân tích dữ liệu
analyzer = DataAnalyzer()
analyzer.check_missing_values(data)

# Xử lý dữ liệu
preprocessor = DataPreprocessor()
clean_data = preprocessor.remove_missing_values(data)
```

### Dự đoán trên dữ liệu mới

```python
# Sau khi đã huấn luyện
predictions = pipeline.predict(new_data)
```

## 📊 Kết quả

Mô hình Linear Regression với polynomial features đạt được:

- **R² Score**: ~0.85
- **MAE**: ~30,000 USD
- **MSE**: ~2,500,000,000
- **MAPE**: ~15%

### 📈 Đặc trưng quan trọng

1. **median_income**: Thu nhập trung bình
2. **rooms_per_household**: Số phòng trên hộ gia đình
3. **population_per_household**: Dân số trên hộ gia đình
4. **bedrooms_per_room**: Tỷ lệ phòng ngủ trên tổng số phòng

## 🔧 Mở rộng

### Thêm mô hình mới

```python
from sklearn.ensemble import RandomForestRegressor

# Trong ModelTrainer
rf_model = RandomForestRegressor(n_estimators=100)
trainer.train_model(rf_model, X_train, y_train, "RandomForest")
```

### Thêm metrics đánh giá mới

```python
# Trong ModelEvaluator
def evaluate_model_extended(self, model, X_test, y_test):
    # Thêm các metrics khác như RMSE, Adjusted R², etc.
    pass
```

## 🧪 Testing

Chạy unit tests:

```bash
python -m pytest tests/
```

Hoặc test từng module:

```bash
python -m pytest tests/test_pipeline.py -v
```

## 📈 Roadmap

- [ ] Thêm nhiều algorithms ML (Random Forest, XGBoost, Neural Networks)
- [ ] Hyperparameter tuning tự động
- [ ] Model ensembling
- [ ] Web interface với Flask/FastAPI
- [ ] Docker containerization
- [ ] Model deployment với MLflow
- [ ] Time series analysis cho dự đoán xu hướng giá

⭐ **Nếu project này hữu ích, hãy cho chúng tôi một star!** ⭐
