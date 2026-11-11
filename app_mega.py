import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
import os

# Cấu hình trang web
st.set_page_config(
    page_title="Thống Kê & Gợi Ý Vietlott Mega 6/45",
    page_icon="🎱",
    layout="wide"
)

st.title("🎱 Phân Tích Thống Kê & Gợi Ý Số Vietlott Mega 6/45")
st.warning("⚠️ **Lưu ý:** Xổ số là trò chơi ngẫu nhiên độc lập. Các phân tích và bộ số dưới đây chỉ mang tính chất thống kê tham khảo, không đảm bảo trúng thưởng.")

# Hàm cào dữ liệu mới nhất trực tiếp từ trang web xổ số
def fetch_latest_from_web():
    urls = [
        "https://minhngoc.net.vn",
        "https://ketqua04.net"
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    new_records = []
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                # Tìm các khối chứa kết quả ngày và 6 quả bóng số
                text_blocks = soup.get_text(separator="\n").split("\n")
                
                # Quét tìm định dạng ngày (dd/mm/yyyy) và chuỗi 6 số liên tiếp từ 01-45
                current_date = None
                for line in text_blocks:
                    line = line.strip()
                    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
                    if date_match:
                        current_date = pd.to_datetime(date_match.group(1), format='%d/%m/%Y', errors='coerce')
                    
                    # Tìm dãy 6 số từ 01 đến 45
                    nums = re.findall(r'\b(0[1-9]|[1-3][0-9]|4[0-5])\b', line)
                    if len(nums) == 6 and current_date is not None and not pd.isna(current_date):
                        int_nums = sorted([int(x) for x in nums])
                        if len(set(int_nums)) == 6:
                            new_records.append({
                                'Ngày': current_date.strftime('%Y-%m-%d'),
                                'N1': int_nums[0], 'N2': int_nums[1], 'N3': int_nums[2],
                                'N4': int_nums[3], 'N5': int_nums[4], 'N6': int_nums[5]
                            })
                            current_date = None # Reset sau khi lấy xong bộ số của ngày đó
                if new_records:
                    break
        except Exception:
            continue
    return pd.DataFrame(new_records)

# Hàm tổng hợp dữ liệu từ file gốc + dữ liệu mới từ web
@st.cache_data(ttl=1800) # Tự động làm mới mỗi 30 phút
def load_data():
    df_base = pd.DataFrame()
    source_notes = []

    # 1. Đọc file lịch sử trên GitHub nếu có
    local_file = "power645.jsonl"
    if os.path.exists(local_file):
        try:
            df = pd.read_json(local_file, lines=True)
            df['date'] = pd.to_datetime(df['date'])
            numbers_df = pd.DataFrame(df['result'].tolist(), columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6'])
            numbers_df['Ngày'] = df['date'].dt.strftime('%Y-%m-%d')
            df_base = numbers_df[['Ngày', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
            source_notes.append("File lịch sử GitHub")
        except Exception:
            pass

    # 2. Quét thêm dữ liệu trực tiếp từ Web
    df_web = fetch_latest_from_web()
    if not df_web.empty:
        source_notes.append("Cập nhật trực tiếp từ Web")
        if not df_base.empty:
            df_base = pd.concat([df_base, df_web], ignore_index=True)
        else:
            df_base = df_web

    # 3. Xử lý trùng lặp và sắp xếp nếu có dữ liệu thật
    if not df_base.empty:
        df_base = df_base.drop_duplicates(subset=['Ngày'], keep='last')
        df_base = df_base.sort_values('Ngày').reset_index(drop=True)
        df_base.insert(0, 'Kỳ quay', [f"#{i+1:05d}" for i in range(len(df_base))])
        return df_base, " + ".join(source_notes)

    # 4. Dự phòng nếu cả file lẫn web đều không lấy được
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='2D')
    draws = [sorted(np.random.choice(range(1, 46), 6, replace=False)) for _ in range(500)]
    df_fallback = pd.DataFrame(draws, columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6'])
    df_fallback['Ngày'] = dates.strftime('%Y-%m-%d')
    df_fallback.insert(0, 'Kỳ quay', [f"#{i+1:05d}" for i in range(500)])
    return df_fallback, "Dữ liệu mô phỏng (Không kết nối được Web/File)"

# Nút làm mới dữ liệu thủ công
if st.button("🔄 Cập nhật kết quả mới nhất từ Web"):
    st.cache_data.clear()
    st.rerun()

df, data_source = load_data()
st.caption(f"Nguồn dữ liệu: **{data_source}** | Tổng số kỳ phân tích: **{len(df)} kỳ** | Kỳ gần nhất: **{df['Ngày'].iloc[-1]}**")

# --- PHÂN TÍCH TẦN SUẤT ---
all_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values.flatten()
freq_series = pd.Series(all_numbers).value_counts().reindex(range(1, 46), fill_value=0)
freq_df = pd.DataFrame({'Số': freq_series.index, 'Số lần xuất hiện': freq_series.values})
freq_df['Tỷ lệ (%)'] = (freq_df['Số lần xuất hiện'] / len(df) * 100).round(2)

# Tính số kỳ chưa về (Lô gan)
last_seen = {}
for num in range(1, 46):
    mask = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].isin([num]).any(axis=1)
    indices = df.index[mask].tolist()
    if indices:
        last_seen[num] = len(df) - 1 - indices[-1]
    else:
        last_seen[num] = len(df)

freq_df['Số kỳ chưa về'] = freq_df['Số'].map(last_seen)

# --- GIAO DIỆN CHÍNH ---
tab1, tab2, tab3 = st.tabs(["🎲 Tạo Bộ Số Gợi Ý", "📊 Biểu Đồ Tần Suất", "📋 Lịch Sử Kết Quả"])

with tab1:
    st.subheader("Trình tạo bộ số Mega 6/45 thông minh")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        method = st.radio(
            "Chọn phương pháp tạo số:",
            [
                "Ngẫu nhiên chuẩn (Xác suất đều)",
                "Ưu tiên Số Nóng (Xuất hiện nhiều)",
                "Ưu tiên Số Lạnh (Lâu chưa về)",
                "Cân bằng Chẵn - Lẻ (3 Chẵn / 3 Lẻ)"
            ]
        )
        num_sets = st.slider("Số lượng bộ số muốn tạo:", min_value=1, max_value=10, value=3)
        generate_btn = st.button("🚀 Tạo bộ số may mắn", type="primary", use_container_width=True)

    with col2:
        if generate_btn:
            st.success(f"Đã tạo **{num_sets}** bộ số theo phương pháp: **{method}**")
            for i in range(num_sets):
                if method == "Ngẫu nhiên chuẩn (Xác suất đều)":
                    chosen = sorted(np.random.choice(range(1, 46), 6, replace=False))
                elif method == "Ưu tiên Số Nóng (Xuất hiện nhiều)":
                    weights = freq_df['Số lần xuất hiện'].values + 1
                    probs = weights / weights.sum()
                    chosen = sorted(np.random.choice(range(1, 46), 6, replace=False, p=probs))
                elif method == "Ưu tiên Số Lạnh (Lâu chưa về)":
                    weights = freq_df['Số kỳ chưa về'].values + 1
                    probs = weights / weights.sum()
                    chosen = sorted(np.random.choice(range(1, 46), 6, replace=False, p=probs))
                else: # Cân bằng Chẵn - Lẻ
                    evens = np.random.choice(range(2, 46, 2), 3, replace=False)
                    odds = np.random.choice(range(1, 46, 2), 3, replace=False)
                    chosen = sorted(np.concatenate([evens, odds]))
                
                balls_html = "".join([
                    f"<span style='display:inline-block; background-color:#e60000; color:white; font-weight:bold; "
                    f"border-radius:50%; width:42px; height:42px; line-height:42px; text-align:center; "
                    f"margin-right:8px; font-size:18px;'>{n:02d}</span>"
                    for n in chosen
                ])
                st.markdown(f"**Vé #{i+1}:** &nbsp; {balls_html}", unsafe_allow_html=True)
                st.write("")

with tab2:
    st.subheader("Thống kê tần suất xuất hiện (Từ 01 đến 45)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("🔥 **Top 10 Số Nóng (Về nhiều nhất)**")
        top_hot = freq_df.sort_values(by='Số lần xuất hiện', ascending=False).head(10)
        st.dataframe(top_hot[['Số', 'Số lần xuất hiện', 'Tỷ lệ (%)']], hide_index=True, use_container_width=True)
        
    with c2:
        st.markdown("❄️ **Top 10 Số Lạnh (Lâu chưa về nhất)**")
        top_cold = freq_df.sort_values(by='Số kỳ chưa về', ascending=False).head(10)
        st.dataframe(top_cold[['Số', 'Số kỳ chưa về', 'Số lần xuất hiện']], hide_index=True, use_container_width=True)

    fig = px.bar(
        freq_df, 
        x='Số', 
        y='Số lần xuất hiện',
        color='Số lần xuất hiện',
        color_continuous_scale='Reds',
        title="Biểu đồ phân bổ tần suất 45 con số Mega 6/45"
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Dữ liệu các kỳ quay gần nhất")
    st.dataframe(df.tail(50).iloc[::-1], hide_index=True, use_container_width=True)
