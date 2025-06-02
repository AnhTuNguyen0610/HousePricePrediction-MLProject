import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


class ModelEvaluator:
    """Lớp đánh giá mô hình"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Đánh giá hiệu suất mô hình"""
        y_pred = model.predict(X_test)
        
        # Các chỉ số đánh giá
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model.__class__.__name__}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - MAPE: {mape:.2f}%")
        print("-" * 100)
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape
        }
