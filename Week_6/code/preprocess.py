import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re

# === 讀取 XML 檔案 ===
xml_path = "./O-A0038-003.xml"  # 改成你本地檔案路徑
tree = ET.parse(xml_path)
root = tree.getroot()

# === 擷取 Content 標籤內的溫度格點資料（以逗號分隔）===
content_text = root.find(".//{urn:cwa:gov:tw:cwacommon:0.1}Content").text

# 修復格式錯誤（有些數值之間以 \n 分隔）
cleaned_text = re.sub(r"\n", ",", content_text.strip())
values_str = cleaned_text.split(",")
values = np.array([float(x) for x in values_str if x.strip() != ""])

# === reshape 成 120 (rows) × 67 (cols) ===
grid = values.reshape((120, 67))  # [緯度, 經度]

# === 建立經緯度座標網格 ===
lon_start, lat_start = 120.00, 25.45
lon_step, lat_step = 0.03, -0.03
lons = np.array([lon_start + i * lon_step for i in range(67)])
lats = np.array([lat_start + j * lat_step for j in range(120)])
lon_grid, lat_grid = np.meshgrid(lons, lats)

# === 分類資料集 (Classification) ===
classification_df = pd.DataFrame({
    'lon': np.round(lon_grid.ravel(), 2),
    'lat': np.round(lat_grid.ravel(), 2),
    'label': np.where(grid.ravel() == -999, 0, 1)
})

# === 回歸資料集 (Regression) ===
valid_mask = grid.ravel() != -999
regression_df = pd.DataFrame({
    'lon': np.round(lon_grid.ravel()[valid_mask], 2),
    'lat': np.round(lat_grid.ravel()[valid_mask], 2),
    'value': grid.ravel()[valid_mask]
})

# === 輸出結果預覽 ===
print("📌 分類資料集（前5筆）:")
print(classification_df.head())

print("\n📌 回歸資料集（前5筆）:")
print(regression_df.head())

# 可選：存成 CSV
classification_df.to_csv("classification_data.csv", index=False)
regression_df.to_csv("regression_data.csv", index=False)