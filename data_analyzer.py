import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis



class DataAnalyzer:
    """Lớp phân tích dữ liệu và thống kê mô tả"""
    
    @staticmethod
    def check_missing_values(data):
        """Kiểm tra giá trị thiếu trong dữ liệu"""
        print("KIỂM TRA MISSING VALUE:")
        print("-" * 100)
        print("Tổng số Missing Value theo cột:")
        print("-" * 100)
        print(data.isnull().sum())
        print("-" * 100)
        print("Phần trăm Missing Value theo cột:")
        print("-" * 100)
        print(data.isnull().mean() * 100)
        print("-" * 100)
    
    @staticmethod
    def analyze_distribution(df, col, plot_=False):
        """Phân tích phân phối dữ liệu của một cột"""
        x = df[col]
        skewness = skew(x)
        kurt = kurtosis(x)
        
        print(f"Phân tích cột: {col}")
        print(f"Độ lệch là: {skewness}")
        print(f"Độ nhọn là: {kurt}")
        
        # Gợi ý xử lý độ lệch
        if skewness > 3:
            sug = 'Lệch phải mạnh -> Dùng log hoặc sqrt hoặc **0.3'
        elif skewness > 1.5:
            sug = 'Lệch phải nhẹ -> Dùng sqrt'
        elif skewness < -3:
            sug = 'Lệch trái mạnh -> Dùng log hoặc **0.3 hoặc sqrt'
        elif skewness < -1.5:
            sug = 'Lệch trái nhẹ -> dùng sqrt'
        else:
            sug = ' Dữ liệu khá cân đối -> có thể giữ nguyên'
        
        print(sug)
        
        # Gợi ý xử lý độ nhọn
        if kurt > 4:
            sug1 = 'Đỉnh nhọn -> có thể có outlier -> nên kiểm tra ngoại lệ!'
        elif kurt < 2:
            sug1 = 'Đỉnh bẹt -> Dữ liệu phân tán, có thể chuẩn hóa'
        else:
            sug1 = 'Dữ liệu Phân phối chuẩn'
        
        print(sug1)
        print("-" * 100)
        
        # Vẽ biểu đồ nếu được yêu cầu
        if plot_:
            plt.figure(figsize=(10, 4))
            sns.histplot(x, kde=True, bins=30)
            plt.title(f"Độ lệch của {col} là {skewness:.3f}")
            plt.xlabel(col)
            plt.show()
        
        return skewness, kurt
