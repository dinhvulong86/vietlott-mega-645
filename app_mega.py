import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Phân Tích & Dự Đoán Vietlott Mega 6/45", page_icon="🎱", layout="wide")
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
10/06/2026,03 - 09 - 17 - 22 - 27 - 32
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

if 'history_text' not in st.session_state: st.session_state.history_text = RAW_HISTORY_DATA

st.sidebar.header("📥 Cập nhật dữ liệu mới")
uploaded_file = st.sidebar.file_uploader("Tải lên file lịch sử (.txt / .csv)", type=['txt', 'csv'])
if uploaded_file is not None: st.session_state.history_text = uploaded_file.getvalue().decode("utf-8"); st.sidebar.success("Đã nạp dữ liệu từ file tải lên!")

st.sidebar.divider()
st.sidebar.subheader("➕ Thêm nhanh 1 kỳ quay mới")
new_date = st.sidebar.text_input("Ngày quay (dd/mm/yyyy):", placeholder="Ví dụ: 11/11/2026")
new_nums = st.sidebar.text_input("6 số trúng thưởng:", placeholder="Ví dụ: 05 - 12 - 19 - 24 - 33 - 41")
if st.sidebar.button("Thêm vào danh sách", use_container_width=True): st.session_state.history_text += f"\n{new_date.strip()},{new_nums.strip()}" if (new_date and new_nums) else ""; st.sidebar.success("Đã cập nhật!") if (new_date and new_nums) else st.sidebar.error("Vui lòng nhập đủ ngày và 6 số.")

lines = [l.strip() for l in st.session_state.history_text.strip().splitlines() if ',' in l]
records = []
for l in lines:
    p = l.split(',')
    ns = sorted([int(x.strip()) for x in p.split('-') if x.strip().isdigit()])
    dt = pd.to_datetime(p.strip(), format='%d/%m/%Y', errors='coerce')
    if len(ns) == 6 and not pd.isna(dt): records.append({'dt': dt, 'Ngày': dt.strftime('%d/%m/%Y'), 'N1': ns, 'N2': ns, 'N3': ns, 'N4': ns, 'N5': ns, 'N6': ns, 'Tổng': sum(ns)})

df = pd.DataFrame(records).drop_duplicates(subset=['Ngày'], keep='last').sort_values('dt').reset_index(drop=True).drop(columns=['dt'])
df.insert(0, 'Kỳ', [f"#{i+1:03d}" for i in range(len(df))])

draws = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].values
t_mat = np.ones((46, 46))
for i in range(len(draws) - 1): np.add.at(t_mat, (draws[i][:, None], draws[i + 1][None, :]), 1)
latest_draw = draws[-1]
markov_probs = t_mat[latest_draw].sum(axis=0)[1:46]
markov_probs = markov_probs / markov_probs.sum()
latest_str = " - ".join([f"{x:02d}" for x in latest_draw])

st.caption(f"Nguồn dữ liệu: **Thực tế Mega 6/45 ({len(df)} kỳ: {df['Ngày'].iloc} → {df['Ngày'].iloc[-1]})** | Kỳ gần nhất: **{latest_str}**")

freq_series = pd.Series(draws.flatten()).value_counts().reindex(range(1, 46), fill_value=0)
freq_df = pd.DataFrame({'Số': freq_series.index, 'Số lần xuất hiện': freq_series.values})
freq_df['Tỷ lệ (%)'] = (freq_df['Số lần xuất hiện'] / len(df) * 100).round(2)
freq_df['Xác suất Markov (%)'] = (markov_probs * 100).round(2)
freq_df['Số kỳ chưa về'] = [len(df) - 1 - df.index[df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].isin([n]).any(axis=1)].tolist()[-1] if df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6']].isin([n]).any(axis=1).any() else len(df) for n in range(1, 46)]

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Dự Đoán Chuỗi Markov", "📊 Phân Tích Chuyên Sâu", "🔍 Dò Vé Lịch Sử", "📋 Bảng Kết Quả"])

tab1.subheader("Mô hình Dự đoán Xác suất Chuyển tiếp (Markov + Bộ lọc chuẩn)")
col1, col2 = tab1.columns([1.1, 1.4])
algo_choice = col1.radio("Chọn thuật toán phân tích:", ["Chuỗi Markov (Dựa theo kỳ gần nhất) + Bộ lọc chuẩn", "Kết hợp Markov + Tần suất Số Nóng + Bộ lọc chuẩn", "Hồi quy Số Lạnh (Lô gan) + Bộ lọc chuẩn", "Mô phỏng Monte Carlo thuần túy + Bộ lọc chuẩn"])
sum_range = col1.slider("Giới hạn Tổng điểm 6 số (Vùng chuẩn: 105 - 175):", min_value=50, max_value=240, value=(105, 175))
balance_eo = col1.checkbox("Lọc cân bằng Chẵn/Lẻ (Tỷ lệ 2-4, 3-3 hoặc 4-2)", value=True)
num_sets = col1.slider("Số lượng bộ số đề xuất:", min_value=1, max_value=10, value=5)
predict_btn = col1.button("🚀 Chạy mô hình dự đoán", type="primary", use_container_width=True)

if predict_btn:
    col2.success(f"Kết quả phân tích từ kỳ gốc **{latest_str}** (Tổng điểm lọc: {sum_range}–{sum_range}):")
    hot_p = (freq_df['Số lần xuất hiện'].values + 1) / (freq_df['Số lần xuất hiện'].values + 1).sum()
    cold_p = (freq_df['Số kỳ chưa về'].values + 1) / (freq_df['Số kỳ chưa về'].values + 1).sum()
    weights = {"Chuỗi Markov (Dựa theo kỳ gần nhất) + Bộ lọc chuẩn": markov_probs, "Kết hợp Markov + Tần suất Số Nóng + Bộ lọc chuẩn": 0.6 * markov_probs + 0.4 * hot_p, "Hồi quy Số Lạnh (Lô gan) + Bộ lọc chuẩn": cold_p, "Mô phỏng Monte Carlo thuần túy + Bộ lọc chuẩn": np.ones(45) / 45.0}[algo_choice]
    candidates = [sorted(np.random.choice(range(1, 46), 6, replace=False, p=weights)) for _ in range(4000)]
    valid_list = []
    for c in candidates:
        ev = sum(1 for x in c if x % 2 == 0)
        max_seq = max(len(list(g)) for _, g in pd.Series(c).groupby(pd.Series(c).diff().ne(1).cumsum()))
        if (sum_range <= sum(c) <= sum_range) and (not balance_eo or 2 <= ev <= 4) and (max_seq < 4) and (c not in valid_list): valid_list.append(c)
        if len(valid_list) >= num_sets: break
    for idx, combo in enumerate(valid_list):
        evens = sum(1 for x in combo if x % 2 == 0)
        balls_html = "".join([f"<span style='display:inline-block; background-color:#d90429; color:white; font-weight:bold; border-radius:50%; width:40px; height:40px; line-height:40px; text-align:center; margin-right:6px; font-size:17px;'>{n:02d}</span>" for n in combo])
        col2.markdown(f"**Đề xuất #{idx+1}:** &nbsp; {balls_html} &nbsp; <span style='color:gray; font-size:14px;'>(Tổng: <b>{sum(combo)}</b> | {6-evens} Lẻ - {evens} Chẵn)</span>", unsafe_allow_html=True)
        col2.write("")

tab2.subheader("Phân tích Xác suất Markov & Phân phối Tổng điểm")
c1, c2, c3 = tab2.columns(3)
c1.markdown("🧠 **Top 10 Số có xác suất Markov cao nhất kỳ tới**")
c1.dataframe(freq_df.sort_values(by='Xác suất Markov (%)', ascending=False).head(10)[['Số', 'Xác suất Markov (%)', 'Số kỳ chưa về']], hide_index=True, use_container_width=True)
c2.markdown("🔥 **Top 10 Số Nóng (Về nhiều nhất lịch sử)**")
c2.dataframe(freq_df.sort_values(by='Số lần xuất hiện', ascending=False).head(10)[['Số', 'Số lần xuất hiện', 'Tỷ lệ (%)']], hide_index=True, use_container_width=True)
c3.markdown("❄️ **Top 10 Số Lạnh (Lâu chưa về nhất)**")
c3.dataframe(freq_df.sort_values(by='Số kỳ chưa về', ascending=False).head(10)[['Số', 'Số kỳ chưa về', 'Số lần xuất hiện']], hide_index=True, use_container_width=True)

fig_sum = px.histogram(df, x='Tổng', nbins=15, title=f"Phân phối Tổng điểm 6 số trong {len(df)} kỳ lịch sử (Trung bình: {df['Tổng'].mean():.1f} điểm)", color_discrete_sequence=['#ef233c'])
tab2.plotly_chart(fig_sum, use_container_width=True)
fig_freq = px.bar(freq_df, x='Số', y='Xác suất Markov (%)', color='Xác suất Markov (%)', color_continuous_scale='Reds', title=f"Biểu đồ trọng số xác suất Chuỗi Markov cho kỳ tiếp theo (Sau kỳ {latest_str})")
fig_freq.update_xaxes(tickmode='linear', tick0=1, dtick=1)
tab2.plotly_chart(fig_freq, use_container_width=True)
tab3.subheader("Kiểm thử bộ số của bạn với toàn bộ lịch sử")
user_input = tab3.text_input("Nhập 6 con số của bạn (cách nhau bởi dấu cách hoặc dấu gạch ngang):", placeholder="Ví dụ: 04 14 20 27 36 41")
if tab3.button("🔍 Kiểm tra lịch sử trúng thưởng"):
clean_nums = sorted(list(set([int(x) for x in user_input.replace('-', ' ').replace(',', ' ').split() if x.isdigit() and 1 <= int(x) <= 45])))
user_set = set(clean_nums)
prize_map = {3: "Giải Ba (30.000đ)", 4: "Giải Nhì (300.000đ)", 5: "Giải Nhất (10.000.000đ)", 6: "JACKPOT"}
matches = [{'Kỳ': r['Kỳ'], 'Ngày': r['Ngày'], 'Kết quả kỳ đó': f"{r['N1']:02d} - {r['N2']:02d} - {r['N3']:02d} - {r['N4']:02d} - {r['N5']:02d} - {r['N6']:02d}", 'Số trùng': f"{len(user_set.intersection({r['N1'], r['N2'], r['N3'], r['N4'], r['N5'], r['N6']}))} số", 'Hạng giải': prize_map[len(user_set.intersection({r['N1'], r['N2'], r['N3'], r['N4'], r['N5'], r['N6']}))]} for _, r in df.iterrows() if len(clean_nums) == 6 and len(user_set.intersection({r['N1'], r['N2'], r['N3'], r['N4'], r['N5'], r['N6']})) >= 3]
tab3.error("Vui lòng nhập chính xác 6 số khác nhau từ 01 đến 45.") if len(clean_nums) != 6 else (tab3.success(f"Bộ số của bạn đã từng trúng giải {len(matches)} lần trong {len(df)} kỳ lịch sử!") or tab3.dataframe(pd.DataFrame(matches).iloc[::-1], hide_index=True, use_container_width=True)) if matches else tab3.info(f"Bộ số này chưa từng trùng từ 3 số trở lên trong {len(df)} kỳ lịch sử vừa qua.")
tab4.subheader("Dữ liệu lịch sử các kỳ quay")
tab4.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
