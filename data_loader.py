import pandas as pd


class DataLoader:
    """Lớp chịu trách nhiệm tải và hiển thị dữ liệu ban đầu"""
    
    def __init__(self, url=None):
        self.url = url or "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
        self.data = None
    
    def load_data(self):
        """Tải dữ liệu từ URL"""
        try:
            self.data = pd.read_csv(self.url)
            print("5 DÒNG ĐẦU CỦA DỮ LIỆU:")
            print("-" * 100)
            print(self.data.head())
            print("-" * 100)
            return self.data
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None