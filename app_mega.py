import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Phân Tích & Dự Đoán Vietlott Mega 6/45 (AI/Markov)",
    page_icon="🎱",
    layout="wide"
)

st.title("🎱 Phân Tích Chuyên Sâu & Dự Đoán Mega 6/45 (Chuỗi Markov)")
st.warning("⚠️ **Lưu ý:** Mô hình sử dụng Chuỗi Markov và Bộ lọc phân phối chuẩn từ dữ liệu lịch sử để tối ưu hóa cấu trúc bộ số. Kết quả chỉ mang tính chất thống kê tham khảo.")

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
                    'N6': nums[5],
                    'Tổng': sum(nums)
                })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=['Ngày'], keep='last')
        df = df.sort_values('dt').reset_index(drop=True)
        df = df.drop(columns=['dt'])
        df.insert(0, 'Kỳ', [f"#{i+1:03d}" for i in range(len(df))])
    return df

def build_markov_probabilities(df):
    transition_matrix = np.ones((46, 46))
    draws = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
    for i in range(len(draws) - 1):
        current_draw = draws[i]
        next_draw = draws[i + 1]
        for num_curr in current_draw:
            for num_next in next_draw:
                transition_matrix[num_curr][num_next] += 1
    
    last_draw = draws[-1]
    next_scores = np.zeros(46)
    for num in last_draw:
        next_scores += transition_matrix[num]
    
    probs = next_scores[1:46]
    probs = probs / probs.sum()
    return probs, last_draw

def is_valid_combination(combo, min_sum, max_sum, balance_even_odd):
    total = sum(combo)
    if total < min_sum or total > max_sum:
        return False
    if balance_even_odd:
        evens = sum(1 for x in combo if x % 2 == 0)
        if evens < 2 or evens > 4:
            return False
    consecutive = 1
    for i in range(1, len(combo)):
        if combo[i] == combo[i-1] + 1:
            consecutive += 1
            if consecutive >= 4:
                return False
        else:
            consecutive = 1
    return True

if 'history_text' not in st.session_state:
    st.session_state.history_text = RAW_HISTORY_DATA

st.sidebar.header("📥 Cập nhật dữ liệu mới")
uploaded_file = st.sidebar.file_uploader("Tải lên file lịch sử (.txt / .csv)", type=['txt', 'csv'])
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
markov_probs, latest_draw = build_markov_probabilities(df)
latest_str = " - ".join([f"{x:02d}" for x in latest_draw])

st.caption(f"Nguồn dữ liệu: **Thực tế Mega 6/45 ({len(df)} kỳ: {first_day} → {last_day})** | Kỳ gần nhất: **{latest_str}**")

all_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values.flatten()
freq_series = pd.Series(all_numbers).value_counts().reindex(range(1, 46), fill_value=0)
freq_df = pd.DataFrame({'Số': freq_series.index, 'Số lần xuất hiện': freq_series.values})
freq_df['Tỷ lệ (%)'] = (freq_df['Số lần xuất hiện'] / len(df) * 100).round(2)
freq_df['Xác suất Markov (%)'] = (markov_probs * 100).round(2)

last_seen = {}
for num in range(1, 46):
    mask = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].isin([num]).any(axis=1)
    indices = df.index[mask].tolist()
    if indices:
        last_seen[num] = len(df) - 1 - indices[-1]
    else:
        last_seen[num] = len(df)
freq_df['Số kỳ chưa về'] = freq_df['Số'].map(last_seen)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Dự Đoán Chuỗi Markov",
    "📊 Phân Tích Chuyên Sâu",
    "🔍 Dò Vé Lịch Sử",
    "📋 Bảng Kết Quả"
])

tab1.subheader("Mô hình Dự đoán Xác suất Chuyển tiếp (Markov + Bộ lọc chuẩn)")
col1, col2 = tab1.columns([1.1, 1.4])

algo_choice = col1.radio(
    "Chọn thuật toán phân tích:",
    [
        "Chuỗi Markov (Dựa theo kỳ gần nhất) + Bộ lọc chuẩn",
        "Kết hợp Markov + Tần suất Số Nóng + Bộ lọc chuẩn",
        "Hồi quy Số Lạnh (Lô gan) + Bộ lọc chuẩn",
        "Mô phỏng Monte Carlo thuần túy + Bộ lọc chuẩn"
    ]
)

sum_range = col1.slider(
    "Giới hạn Tổng điểm 6 số (Vùng chuẩn: 105 - 175):",
    min_value=50,
    max_value=240,
    value=(105, 175)
)
balance_eo = col1.checkbox("Lọc cân bằng Chẵn/Lẻ (Tỷ lệ 2-4, 3-3 hoặc 4-2)", value=True)
num_sets = col1.slider("Số lượng bộ số đề xuất:", min_value=1, max_value=10, value=5)
predict_btn = col1.button("🚀 Chạy mô hình dự đoán", type="primary", use_container_width=True)

if predict_btn:
    col2.success(f"Kết quả phân tích từ kỳ gốc **{latest_str}** (Tổng điểm lọc: {sum_range[0]}–{sum_range[1]}):")
    
    if algo_choice == "Chuỗi Markov (Dựa theo kỳ gần nhất) + Bộ lọc chuẩn":
        weights = markov_probs
    elif algo_choice == "Kết hợp Markov + Tần suất Số Nóng + Bộ lọc chuẩn":
        hot_probs = (freq_df['Số lần xuất hiện'].values + 1)
        hot_probs = hot_probs / hot_probs.sum()
        weights = 0.6 * markov_probs + 0.4 * hot_probs
    elif algo_choice == "Hồi quy Số Lạnh (Lô gan) + Bộ lọc chuẩn":
        cold_probs = (freq_df['Số kỳ chưa về'].values + 1)
        weights = cold_probs / cold_probs.sum()
    else:
weights = np.ones(45) / 45.0
generated_list = []
attempts = 0
while len(generated_list) < num_sets and attempts < 5000:
attempts += 1
candidate = sorted(np.random.choice(range(1, 46), 6, replace=False, p=weights))
if is_valid_combination(candidate, sum_range[0], sum_range[1], balance_eo):
if candidate not in generated_list:
generated_list.append(candidate)
if not generated_list:
col2.error("Điều kiện lọc quá khắt khe, không tìm thấy bộ số phù hợp. Hãy nới rộng khoảng Tổng điểm!")
else:
for idx, combo in enumerate(generated_list):
evens = sum(1 for x in combo if x % 2 == 0)
odds = 6 - evens
total_pts = sum(combo)
balls_html = "".join([
f"{n:02d}"
for n in combo
])
col2.markdown(
f"Đề xuất #{idx+1}:   {balls_html}   "
f"(Tổng: {total_pts} | {odds} Lẻ - {evens} Chẵn)",
unsafe_allow_html=True
)
col2.write("")
tab2.subheader("Phân tích Xác suất Markov & Phân phối Tổng điểm")
c1, c2, c3 = tab2.columns(3)
c1.markdown("🧠 Top 10 Số có xác suất Markov cao nhất kỳ tới")
top_markov = freq_df.sort_values(by='Xác suất Markov (%)', ascending=False).head(10)
c1.dataframe(top_markov[['Số', 'Xác suất Markov (%)', 'Số kỳ chưa về']], hide_index=True, use_container_width=True)
c2.markdown("🔥 Top 10 Số Nóng (Về nhiều nhất lịch sử)")
top_hot = freq_df.sort_values(by='Số lần xuất hiện', ascending=False).head(10)
c2.dataframe(top_hot[['Số', 'Số lần xuất hiện', 'Tỷ lệ (%)']], hide_index=True, use_container_width=True)
c3.markdown("❄️ Top 10 Số Lạnh (Lâu chưa về nhất)")
top_cold = freq_df.sort_values(by='Số kỳ chưa về', ascending=False).head(10)
c3.dataframe(top_cold[['Số', 'Số kỳ chưa về', 'Số lần xuất hiện']], hide_index=True, use_container_width=True)
fig_sum = px.histogram(
df,
x='Tổng',
nbins=15,
title=f"Phân phối Tổng điểm 6 số trong {len(df)} kỳ lịch sử (Trung bình: {df['Tổng'].mean():.1f} điểm)",
color_discrete_sequence=['#ef233c']
)
tab2.plotly_chart(fig_sum, use_container_width=True)
fig_freq = px.bar(
freq_df,
x='Số',
y='Xác suất Markov (%)',
color='Xác suất Markov (%)',
color_continuous_scale='Reds',
title=f"Biểu đồ trọng số xác suất Chuỗi Markov cho kỳ tiếp theo (Sau kỳ {latest_str})"
)
fig_freq.update_xaxes(tickmode='linear', tick0=1, dtick=1)
tab2.plotly_chart(fig_freq, use_container_width=True)
tab3.subheader("Kiểm thử bộ số của bạn với toàn bộ lịch sử")
user_input = tab3.text_input("Nhập 6 con số của bạn (cách nhau bởi dấu cách hoặc dấu gạch ngang):", placeholder="Ví dụ: 04 14 20 27 36 41")
if tab3.button("🔍 Kiểm tra lịch sử trúng thưởng"):
clean_nums = sorted(list(set([int(x) for x in user_input.replace('-', ' ').replace(',', ' ').split() if x.isdigit() and 1 <= int(x) <= 45])))
if len(clean_nums) != 6:
tab3.error("Vui lòng nhập chính xác 6 số khác nhau từ 01 đến 45.")
else:
user_set = set(clean_nums)
matches = []
for idx, row in df.iterrows():
draw_set = {row['N1'], row['N2'], row['N3'], row['N4'], row['N5'], row['N6']}
hit_count = len(user_set.intersection(draw_set))
if hit_count >= 3:
prize = {3: "Giải Ba (30.000đ)", 4: "Giải Nhì (300.000đ)", 5: "Giải Nhất (10.000.000đ)", 6: "JACKPOT"}[hit_count]
matches.append({
'Kỳ': row['Kỳ'],
'Ngày': row['Ngày'],
'Kết quả kỳ đó': f"{row['N1']:02d} - {row['N2']:02d} - {row['N3']:02d} - {row['N4']:02d} - {row['N5']:02d} - {row['N6']:02d}",
'Số trùng': f"{hit_count} số",
'Hạng giải': prize
})
if matches:
tab3.success(f"Bộ số của bạn đã từng trúng giải {len(matches)} lần trong {len(df)} kỳ lịch sử!")
tab3.dataframe(pd.DataFrame(matches).iloc[::-1], hide_index=True, use_container_width=True)
else:
tab3.info(f"Bộ số này chưa từng trùng từ 3 số trở lên trong {len(df)} kỳ lịch sử vừa qua.")
tab4.subheader("Dữ liệu lịch sử các kỳ quay")
tab4.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
