import os
import pandas as pd
import re

folder_path = r'./100_case_si_test/processed'

# Lặp qua tất cả các file CSV trong thư mục
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        # Tìm giá trị Ste từ tên file
        match = re.search(r'Ste=(-?\d+)_(\d+)_(\d+)', file)
        if match:
            cond1, cond2, cond3 = match.groups()  # Lấy ba giá trị từ regex
        else:
            cond1, cond2, cond3 = "Unknown", "Unknown", "Unknown"

        file_path = os.path.join(folder_path, file)

        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(file_path)

        # Thêm ba cột mới
        df["cond1"] = cond1
        df["cond2"] = cond2
        df["cond3"] = cond3

        # Ghi đè file CSV với cột mới
        df.to_csv(file_path, index=False)

        print(f"Đã cập nhật: {file} với cond1={cond1}, cond2={cond2}, cond3={cond3}")
