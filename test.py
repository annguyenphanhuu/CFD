import os
import pandas as pd

# Đường dẫn tới thư mục chứa các file CSV
# directory = './100_case_si/processed'
directory = './30_case_duyanh/processed'

# Lặp qua từng file trong thư mục
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Chỉ xử lý các file có đuôi .csv
        file_path = os.path.join(directory, filename)
        
        # Đọc file CSV
        try:
            data = pd.read_csv(file_path)
            column = 'Umean_center'
            if column in data.columns:
                max_value = data[column].max()
                print(f"File: {filename}, Max {column}: {max_value}")
            else:
                print(f"File: {filename} không tồn tại cột")
        
        except Exception as e:
            print(f"Không thể đọc file {filename}. Lỗi: {e}")
