# 🏠 House Price Prediction 

Dự đoán giá nhà bằng Machine Learning.

## 🎯 Giới thiệu

Project này sử dụng bộ dữ liệu California Housing để dự đoán giá nhà dựa trên các đặc trưng như vị trí địa lý, thu nhập trung bình, số phòng, v.v. Được xây dựng theo kiến trúc OOP để dễ dàng bảo trì, mở rộng và tái sử dụng.

## ✨ Tính năng

- 🔄 **Pipeline tự động**: Xử lý dữ liệu từ A-Z
- 📊 **Phân tích dữ liệu**: Phân tích phân phối, missing values, outliers
- 🛠️ **Feature Engineering**: Tạo đặc trưng mới và polynomial features
- 🧹 **Data Cleaning**: Xử lý missing values, outliers, độ lệch phân phối
- 🤖 **Machine Learning**: Linear Regression với khả năng mở rộng
- 📈 **Đánh giá mô hình**: MAE, MSE, R², MAPE
- 🏗️ **Kiến trúc OOP**: Dễ bảo trì và mở rộng

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

- **R² Score**: ~0.80
- **MAE**: ~35,000 USD
- **MSE**: ~2,500,000,000
- **MAPE**: ~20%

### 📈 Đặc trưng quan trọng

1. **median_income**: Thu nhập trung bình
2. **rooms_per_household**: Số phòng trên hộ gia đình
3. **population_per_household**: Dân số trên hộ gia đình
4. **bedrooms_per_room**: Tỷ lệ phòng ngủ trên tổng số phòng


## 📈 Roadmap

- [ ] Thêm nhiều algorithms ML (Random Forest, XGBoost, Neural Networks)
- [ ] Hyperparameter tuning tự động
- [ ] Model ensembling
- [ ] Web interface với Flask/FastAPI
- [ ] Docker containerization
- [ ] Model deployment với MLflow
- [ ] Time series analysis cho dự đoán xu hướng giá
