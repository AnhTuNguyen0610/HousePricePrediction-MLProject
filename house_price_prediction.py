from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from data_loader import DataLoader
from data_analyzer import DataAnalyzer
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator


class HousePricePredictionPipeline:
    """Lớp chính điều phối toàn bộ pipeline"""
    
    def __init__(self, data_url=None):
        self.data_loader = DataLoader(data_url)
        self.analyzer = DataAnalyzer()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        self.raw_data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self.distribution_stats = {}
    
    def run_pipeline(self):
        """Chạy toàn bộ pipeline"""
        # 1. Tải dữ liệu
        print("=== BƯỚC 1: TẢI DỮ LIỆU ===")
        self.raw_data = self.data_loader.load_data()
        if self.raw_data is None:
            return
        
        # 2. Phân tích dữ liệu ban đầu
        print("=== BƯỚC 2: PHÂN TÍCH DỮ LIỆU ===")
        self.analyzer.check_missing_values(self.raw_data)
        
        # 3. Xử lý dữ liệu cơ bản
        print("=== BƯỚC 3: XỬ LÝ DỮ LIỆU CƠ BẢN ===")
        self.processed_data = self.preprocessor.remove_missing_values(self.raw_data)
        self.processed_data = self.preprocessor.encode_categorical(
            self.processed_data, ['ocean_proximity']
        )
        self.processed_data = self.preprocessor.create_features(self.processed_data)
        
        # 4. Phân tích phân phối
        print("=== BƯỚC 4: PHÂN TÍCH PHÂN PHỐI ===")
        print("PHÂN TÍCH PHÂN PHỐI DỮ LIỆU:")
        print("-" * 100)
        
        for col in self.processed_data.columns:
            if 'ocean_proximity' in col:
                continue
            self.distribution_stats[col] = self.analyzer.analyze_distribution(
                self.processed_data, col
            )
        
        # 5. Xử lý phân phối
        print("=== BƯỚC 5: XỬ LÝ PHÂN PHỐI ===")
        print(f"Kích thước bộ dữ liệu trước xử lý: {self.processed_data.shape}")
        print("-" * 100)
        
        self.processed_data = self.preprocessor.fix_distribution(
            self.processed_data, self.distribution_stats
        )
        
        # 6. Chuẩn bị dữ liệu cho mô hình
        print("=== BƯỚC 6: CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH ===")
        self.y = self.processed_data['median_house_value']
        X_features = self.processed_data.drop(columns=['median_house_value'])
        
        # Tạo đặc trưng phi tuyến
        self.X = self.preprocessor.create_polynomial_features(X_features, degree=3)
        self.X = self.preprocessor.handle_nan_after_polynomial(self.X)
        
        # Kiểm tra phân phối của nhãn y
        sk = skew(self.y)
        kt = kurtosis(self.y)
        print(f"Độ lệch của nhãn là: {sk}")
        print(f"Độ nhọn của nhãn là: {kt}")
        print("-" * 100)
        
        # 7. Chia dữ liệu và huấn luyện
        print("=== BƯỚC 7: HUẤN LUYỆN MÔ HÌNH ===")
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            self.X, self.y, scale=False  # Tắt chuẩn hóa để R2 Score cao hơn
        )
        
        # Huấn luyện Linear Regression
        lr_model = LinearRegression()
        self.trainer.train_model(lr_model, X_train, y_train, "LinearRegression")
        
        # 8. Đánh giá mô hình
        print("=== BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH ===")
        print("ĐÁNH GIÁ MÔ HÌNH:")
        print("-" * 100)
        
        results = self.evaluator.evaluate_model(lr_model, X_test, y_test)
        
        return results
    
    def predict(self, new_data):
        """Dự đoán trên dữ liệu mới"""
        if not self.trainer.models:
            print("Chưa có mô hình nào được huấn luyện!")
            return None
        
        # Sử dụng mô hình đầu tiên có sẵn
        model_name = list(self.trainer.models.keys())[0]
        model = self.trainer.models[model_name]
        
        return model.predict(new_data)


# ================================================================
# CHẠY PIPELINE
# ================================================================

if __name__ == "__main__":
    # Khởi tạo và chạy pipeline
    pipeline = HousePricePredictionPipeline()
    results = pipeline.run_pipeline()
    
    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    if results:
        print(f"R² Score: {results['r2']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
        print(f"MAPE: {results['mape']:.2f}%")