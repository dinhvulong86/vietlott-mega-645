import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from io import StringIO

# Cấu hình trang web
st.set_page_config(
    page_title="Thống Kê & Gợi Ý Vietlott Mega 6/45",
    page_icon="🎱",
    layout="wide"
)

st.title("🎱 Phân Tích Thống Kê & Gợi Ý Số Vietlott Mega 6/45")
st.warning("⚠️ **Lưu ý:** Xổ số là trò chơi ngẫu nhiên độc lập. Các phân tích và bộ số dưới đây chỉ mang tính chất thống kê tham khảo, không đảm bảo trúng thưởng.")

# Hàm tải dữ liệu lịch sử Mega 6/45 chuẩn xác
@st.cache_data(ttl=3600)
def load_data():
    # Đường dẫn chính xác tới kho dữ liệu Mega 6/45 (định dạng NDJSON/JSONL)
    url = "https://githubusercontent.com"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_json(StringIO(response.text), lines=True)
            # Sắp xếp dữ liệu theo ngày từ cũ đến mới
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Tách mảng 6 số trúng thưởng thành 6 cột
            numbers_df = pd.DataFrame(df['result'].tolist(), columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6'])
            numbers_df['Ngày'] = df['date'].dt.strftime('%Y-%m-%d')
            numbers_df['Kỳ quay'] = df['id'].astype(str).str.zfill(5)
            
            # Sắp xếp lại thứ tự cột cho dễ nhìn
            numbers_df = numbers_df[['Kỳ quay', 'Ngày', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
            return numbers_df, "Dữ liệu thực tế từ hệ thống Vietlott"
    except Exception as e:
        st.error(f"Lỗi kết nối nguồn dữ liệu: {e}")
    
    # Dự phòng nếu mất mạng hoàn toàn
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='2D')
    draws = [sorted(np.random.choice(range(1, 46), 6, replace=False)) for _ in range(500)]
    df_fallback = pd.DataFrame(draws, columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6'])
    df_fallback['Ngày'] = dates.strftime('%Y-%m-%d')
    df_fallback['Kỳ quay'] = [f"#{i+1:05d}" for i in range(500)]
    df_fallback = df_fallback[['Kỳ quay', 'Ngày', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    return df_fallback, "Dữ liệu mô phỏng (Do lỗi kết nối)"

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
                
                # Hiển thị dạng quả bóng đẹp mắt
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
