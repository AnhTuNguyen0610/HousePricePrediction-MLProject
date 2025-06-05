import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class DataPreprocessor:
    """Lớp xử lý và làm sạch dữ liệu"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = None
    
    def remove_missing_values(self, data):
        """Loại bỏ các dòng có giá trị thiếu"""
        original_shape = data.shape
        data_cleaned = data.dropna()
        print(f"Đã loại bỏ {original_shape[0] - data_cleaned.shape[0]} dòng có missing values")
        return data_cleaned
    
    def encode_categorical(self, data, categorical_columns):
        """Mã hóa biến phân loại"""
        for col in categorical_columns:
            if col in data.columns:
                data = pd.get_dummies(data, columns=[col], drop_first=True)
        
        print('DATA SAU KHI XỬ LÝ BIẾN CATEGORICAL:')
        print("-" * 100)
        print(data.head())
        print("-" * 100)
        return data
    
    def create_features(self, data):
        """Tạo các đặc trưng mới"""
        feature_data = data.copy()
        
        # Thêm các đặc trưng dẫn xuất
        feature_data['rooms_per_household'] = feature_data['total_rooms'] / feature_data['households']
        feature_data['bedrooms_per_room'] = feature_data['total_bedrooms'] / feature_data['total_rooms']
        feature_data['population_per_household'] = feature_data['population'] / feature_data['households']
        feature_data['rooms_per_person'] = feature_data['total_rooms'] / feature_data['population']
        feature_data['bedrooms_per_household'] = feature_data['total_bedrooms'] / feature_data['households']
        feature_data['bedroom_to_income_ratio'] = feature_data['total_bedrooms'] / feature_data['median_income']
        
        print("ĐÃ THÊM ĐẶC TRƯNG DỮ LIỆU!")
        print("-" * 100)
        
        return feature_data
    
    def remove_outliers_iqr(self, df, column):
        """Loại bỏ outliers sử dụng phương pháp IQR"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        print(f"📌 Đã loại bỏ {len(df) - len(filtered_df)} outliers ở cột '{column}'")
        return filtered_df
    
    def fix_distribution(self, data, distribution_stats):
        """Xử lý phân phối dữ liệu dựa trên thống kê"""
        processed_data = data.copy()
        
        for col in distribution_stats.keys():
            if col not in processed_data.columns:
                continue
                
            skewness, kurt = distribution_stats[col]
            
            # Xử lý độ lệch
            if abs(skewness) > 5:
                processed_data[col] = np.log1p(processed_data[col])
            elif abs(skewness) > 3:
                processed_data[col] = processed_data[col] ** 0.3
            elif abs(skewness) > 1:
                processed_data[col] = processed_data[col] ** 0.5
            
            # Xử lý độ nhọn
            if kurt > 5:
                processed_data = self.remove_outliers_iqr(processed_data, col)
            elif kurt < -5:
                scaler = StandardScaler()
                processed_data[col] = scaler.fit_transform(processed_data[[col]]).flatten()
        
        return processed_data
    
    def create_polynomial_features(self, X, degree=3):
        """Tạo đặc trưng phi tuyến"""
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        
        # Chuyển về DataFrame
        feature_names = self.poly.get_feature_names_out()
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
        
        return X_poly_df
    
    def handle_nan_after_polynomial(self, X):
        """Xử lý NaN sau khi tạo đặc trưng phi tuyến"""
        # Xóa các cột có toàn bộ là NaN
        X_clean = X.dropna(axis=1, how='all')
        
        print("-" * 100)
        print("Kích thước bộ dữ liệu sau khi xóa các cột all NaN")
        print(X_clean.shape)
        
        # Xử lý NaN còn lại
        cols_with_nan = X_clean.columns[X_clean.isna().any()]
        print("-" * 100)
        print("Cột còn thiếu giá trị:", cols_with_nan)
        
        for col in cols_with_nan:
            if X_clean[col].dtype in ['float64', 'int64']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0])
        
        print("-" * 100)
        print("Kích thước sau khi xử lý NaN lần cuối:")
        print(X_clean.shape)
        print("-" * 100)
        
        return X_clean