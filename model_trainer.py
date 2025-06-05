from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """Lớp huấn luyện mô hình"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42, scale=False):
        """Chuẩn bị dữ liệu train/test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        print("Kiểm tra bộ dữ liệu lần cuối:")
        print("Số dòng dữ liệu train:", X_train.shape[0])
        print("Số dòng dữ liệu test:", X_test.shape[0])
        print("Số đặc trưng:", X_train.shape[1])
        print("-" * 100)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train, model_name=None):
        """Huấn luyện một mô hình"""
        if model_name is None:
            model_name = model.__class__.__name__
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        
        print(f"Đã huấn luyện mô hình: {model_name}")
        return model
