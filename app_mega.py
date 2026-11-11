import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Cấu hình trang web
st.set_page_config(
    page_title="Thống Kê & Gợi Ý Vietlott Mega 6/45",
    page_icon="🎱",
    layout="wide"
)

st.title("🎱 Phân Tích Thống Kê & Gợi Ý Số Vietlott Mega 6/45")
st.warning("⚠️ **Lưu ý:** Xổ số là trò chơi ngẫu nhiên độc lập. Các phân tích và bộ số dưới đây chỉ mang tính chất thống kê tham khảo, không đảm bảo trúng thưởng.")

# Dữ liệu lịch sử thực tế Mega 6/45 (Từ 01/05/2026 đến 08/11/2026)
RAW_HISTORY_DATA = """01/05/2026,04 - 21 - 25 - 31 - 41 - 43
03/05/2026,07 - 09 - 14 - 24 - 38 - 39
06/05/2026,09 - 12 - 28 - 36 - 37 - 40
08/05/2026,04 - 16 - 17 - 26 - 34 - 42
10/05/2026,03 - 04 - 10 - 11 - 27 - 42
13/05/2026,08 - 10 - 15 - 32 - 35 - 43
15/05/2026,06 - 14 - 20 - 22 - 23 - 44
17/05/2026,04 - 08 - 13 - 15 - 27 - 31
20/05/2026,16 - 28 - 29 - 37 - 40 - 45
22/05/2026,02 - 03 - 14 - 28 - 41 - 44
24/05/2026,04 - 08 - 09 - 14 - 25 - 42
27/05/2026,02 - 13 - 17 - 18 - 31 - 37
29/05/2026,13 - 14 - 24 - 28 - 30 - 42
31/05/2026,01 - 12 - 16 - 21 - 24 - 41
03/06/2026,03 - 11 - 26 - 33 - 36 - 38
05/06/2026,13 - 16 - 19 - 32 - 36 - 39
07/06/2026,14 - 21 - 26 - 30 - 34 - 35
10/06/2026,03 - 09 - 17 - 22 - 32 - 27
12/06/2026,05 - 15 - 30 - 34 - 37 - 38
14/06/2026,07 - 16 - 20 - 22 - 24 - 38
17/06/2026,05 - 06 - 16 - 28 - 35 - 39
19/06/2026,06 - 09 - 19 - 29 - 30 - 36
21/06/2026,03 - 08 - 19 - 27 - 41 - 45
24/06/2026,03 - 11 - 17 - 19 - 30 - 32
26/06/2026,09 - 14 - 31 - 36 - 41 - 45
28/06/2026,02 - 11 - 18 - 33 - 42 - 45
01/07/2026,20 - 24 - 25 - 29 - 40 - 44
03/07/2026,06 - 20 - 21 - 28 - 30 - 39
05/07/2026,01 - 05 - 13 - 29 - 32 - 35
08/07/2026,13 - 14 - 22 - 26 - 37 - 44
10/07/2026,09 - 17 - 23 - 26 - 42 - 44
15/07/2026,07 - 11 - 29 - 37 - 43 - 45
17/07/2026,09 - 18 - 30 - 31 - 39 - 45
19/07/2026,12 - 22 - 24 - 26 - 31 - 37
22/07/2026,03 - 32 - 33 - 34 - 39 - 43
24/07/2026,12 - 16 - 36 - 38 - 41 - 45
26/07/2026,13 - 16 - 27 - 33 - 41 - 44
29/07/2026,02 - 06 - 21 - 31 - 36 - 45
31/07/2026,06 - 16 - 24 - 25 - 38 - 43
02/08/2026,03 - 12 - 20 - 25 - 27 - 37
05/08/2026,02 - 06 - 11 - 16 - 28 - 39
07/08/2026,02 - 08 - 19 - 30 - 36 - 43
09/08/2026,03 - 17 - 20 - 27 - 31 - 35
12/08/2026,15 - 17 - 22 - 29 - 33 - 40
14/08/2026,07 - 09 - 13 - 31 - 35 - 44
16/08/2026,06 - 07 - 15 - 19 - 36 - 41
19/08/2026,06 - 15 - 18 - 33 - 40 - 43
21/08/2026,07 - 26 - 31 - 38 - 43 - 45
23/08/2026,04 - 16 - 17 - 22 - 32 - 39
26/08/2026,03 - 10 - 11 - 16 - 33 - 40
28/08/2026,03 - 13 - 15 - 22 - 36 - 39
30/08/2026,01 - 03 - 12 - 15 - 37 - 45
02/09/2026,06 - 09 - 27 - 29 - 35 - 44
04/09/2026,16 - 21 - 23 - 29 - 34 - 45
06/09/2026,09 - 14 - 22 - 26 - 27 - 40
09/09/2026,12 - 17 - 20 - 21 - 36 - 43
11/09/2026,14 - 18 - 20 - 21 - 26 - 27
13/09/2026,04 - 12 - 31 - 34 - 38 - 41
16/09/2026,01 - 03 - 10 - 11 - 18 - 23
18/09/2026,07 - 12 - 26 - 27 - 41 - 43
20/09/2026,10 - 15 - 16 - 27 - 33 - 38
23/09/2026,02 - 06 - 12 - 15 - 25 - 26
25/09/2026,03 - 15 - 18 - 28 - 33 - 36
27/09/2026,02 - 04 - 13 - 25 - 31 - 39
30/09/2026,02 - 09 - 24 - 27 - 30 - 33
02/10/2026,01 - 06 - 20 - 27 - 31 - 41
04/10/2026,15 - 20 - 29 - 37 - 40 - 45
07/10/2026,10 - 14 - 36 - 37 - 41 - 43
09/10/2026,05 - 07 - 12 - 23 - 32 - 41
11/10/2026,03 - 09 - 16 - 29 - 30 - 44
14/10/2026,06 - 16 - 19 - 21 - 34 - 43
16/10/2026,09 - 20 - 26 - 35 - 36 - 38
18/10/2026,10 - 11 - 31 - 38 - 40 - 44
21/10/2026,04 - 10 - 14 - 22 - 25 - 44
23/10/2026,02 - 05 - 11 - 22 - 32 - 40
25/10/2026,04 - 13 - 22 - 25 - 40 - 41
28/10/2026,01 - 09 - 11 - 28 - 33 - 42
30/10/2026,07 - 21 - 22 - 40 - 41 - 45
01/11/2026,05 - 09 - 17 - 18 - 34 - 36
04/11/2026,07 - 14 - 20 - 30 - 34 - 40
06/11/2026,01 - 10 - 37 - 38 - 39 - 45
08/11/2026,01 - 02 - 14 - 20 - 23 - 36"""

# Hàm chuyển đổi định dạng dữ liệu
def parse_vietlott_text(raw_text):
    records = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or ',' not in line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        nums_str = parts[1].strip()
        
        nums = sorted([int(x.strip()) for x in nums_str.split('-') if x.strip().isdigit()])
        if len(nums) == 6:
            dt = pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
            if not pd.isna(dt):
                records.append({
                    'dt': dt,
                    'Ngày': dt.strftime('%d/%m/%Y'),
                    'N1': nums[0],
                    'N2': nums[1],
                    'N3': nums[2],
                    'N4': nums[3],
                    'N5': nums[4],
                    'N6': nums[5]
                })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=['Ngày'], keep='last')
        df = df.sort_values('dt').reset_index(drop=True)
        df = df.drop(columns=['dt'])
        df.insert(0, 'Kỳ', [f"#{i+1:03d}" for i in range(len(df))])
    return df

# Khởi tạo dữ liệu trong session_state
if 'history_text' not in st.session_state:
    st.session_state.history_text = RAW_HISTORY_DATA

# Thanh bên (Sidebar) cập nhật dữ liệu mới
st.sidebar.header("📥 Cập nhật dữ liệu mới")
st.sidebar.write("Tải lên file `.txt` / `.csv` hoặc nhập trực tiếp kỳ quay mới:")

uploaded_file = st.sidebar.file_uploader("Tải lên file lịch sử của bạn", type=['txt', 'csv'])
if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    st.session_state.history_text = content
    st.sidebar.success("Đã nạp dữ liệu từ file tải lên!")

st.sidebar.divider()
st.sidebar.subheader("➕ Thêm nhanh 1 kỳ quay mới")
new_date = st.sidebar.text_input("Ngày quay (dd/mm/yyyy):", placeholder="Ví dụ: 11/11/2026")
new_nums = st.sidebar.text_input("6 số trúng thưởng:", placeholder="Ví dụ: 05 - 12 - 19 - 24 - 33 - 41")
if st.sidebar.button("Thêm vào danh sách", use_container_width=True):
    if new_date and new_nums:
        new_line = f"\n{new_date.strip()},{new_nums.strip()}"
        st.session_state.history_text += new_line
        st.sidebar.success(f"Đã thêm kỳ ngày {new_date}!")
    else:
        st.sidebar.error("Vui lòng nhập đủ ngày và 6 số.")

df = parse_vietlott_text(st.session_state.history_text)

if df.empty:
    st.error("Không có dữ liệu hợp lệ để phân tích. Vui lòng kiểm tra lại định dạng dữ liệu.")
    st.stop()

first_day = df['Ngày'].iloc[0]
last_day = df['Ngày'].iloc[-1]
st.caption(f"Nguồn dữ liệu: **Dữ liệu thực tế Mega 6/45** | Tổng số kỳ phân tích: **{len(df)} kỳ** (Từ **{first_day}** đến **{last_day}**)")

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

# TAB 1: TẠO BỘ SỐ GỢI Ý
tab1.subheader("Trình tạo bộ số Mega 6/45 thông minh")
col1, col2 = tab1.columns([1, 1.5])

method = col1.radio(
    "Chọn phương pháp tạo số:",
    [
        "Ngẫu nhiên chuẩn (Xác suất đều)",
        "Ưu tiên Số Nóng (Xuất hiện nhiều)",
        "Ưu tiên Số Lạnh (Lâu chưa về)",
        "Cân bằng Chẵn - Lẻ (3 Chẵn / 3 Lẻ)"
    ]
)
num_sets = col1.slider("Số lượng bộ số muốn tạo:", min_value=1, max_value=10, value=3)
generate_btn = col1.button("🚀 Tạo bộ số may mắn", type="primary", use_container_width=True)

if generate_btn:
    col2.success(f"Đã tạo **{num_sets}** bộ số theo phương pháp: **{method}**")
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
        else:
            evens = np.random.choice(range(2, 46, 2), 3, replace=False)
            odds = np.random.choice(range(1, 46, 2), 3, replace=False)
            chosen = sorted(np.concatenate([evens, odds]))
        
        balls_html = "".join([
            f"<span style='display:inline-block; background-color:#e60000; color:white; font-weight:bold; "
            f"border-radius:50%; width:42px; height:42px; line-height:42px; text-align:center; "
            f"margin-right:8px; font-size:18px;'>{n:02d}</span>"
            for n in chosen
        ])
        col2.markdown(f"**Vé #{i+1}:** &nbsp; {balls_html}", unsafe_allow_html=True)
        col2.write("")

# TAB 2: BIỂU ĐỒ TẦN SUẤT
tab2.subheader("Thống kê tần suất xuất hiện (Từ 01 đến 45)")
c1, c2 = tab2.columns(2)

c1.markdown("🔥 **Top 10 Số Nóng (Về nhiều nhất)**")
top_hot = freq_df.sort_values(by='Số lần xuất hiện', ascending=False).head(10)
c1.dataframe(top_hot[['Số', 'Số lần xuất hiện', 'Tỷ lệ (%)']], hide_index=True, use_container_width=True)

c2.markdown("❄️ **Top 10 Số Lạnh (Lâu chưa về nhất)**")
top_cold = freq_df.sort_values(by='Số kỳ chưa về', ascending=False).head(10)
c2.dataframe(top_cold[['Số', 'Số kỳ chưa về', 'Số lần xuất hiện']], hide_index=True, use_container_width=True)
fig = px.bar(
freq_df,
x='Số',
y='Số lần xuất hiện',
color='Số lần xuất hiện',
color_continuous_scale='Reds',
title="Biểu đồ phân bổ tần suất 45 con số Mega 6/45"
)
fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
tab2.plotly_chart(fig, use_container_width=True)

TAB 3: LỊCH SỬ KẾT QUẢ

tab3.subheader("Dữ liệu các kỳ quay gần nhất")
tab3.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
