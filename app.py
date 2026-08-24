import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 세션 상태 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="2D 도선 자기장 시각화 플랫폼", layout="wide")

if "wires" not in st.session_state:
    st.session_state.wires = []

if "points" not in st.session_state:
    st.session_state.points = [
        {"name": "p", "x": -1.0, "y": 0.0},
        {"name": "O", "x": 0.0, "y": 0.0},
        {"name": "q", "x": 1.0, "y": 0.0}
    ]

if "symbol_values" not in st.session_state:
    st.session_state.symbol_values = {"I_0": 1.0}

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# 교재 문제 예시 불러오기 함수
def load_preset_problem():
    st.session_state.wires = [
        {"type": "straight", "name": "A", "p1": (-2.0, -4.0), "p2": (-2.0, 4.0), "current_symbol": "I_0", "direction": -1}, # 아래 방향
        {"type": "straight", "name": "B", "p1": (2.0, -4.0), "p2": (2.0, 4.0), "current_symbol": "I_0", "direction": 1},   # 위 방향
        {"type": "straight", "name": "C", "p1": (-4.0, -2.0), "p2": (4.0, -2.0), "current_symbol": "I_0", "direction": 1},  # 오른쪽 방향
        {"type": "circle", "name": "D", "center": (0.0, -1.0), "radius": 1.0, "current_symbol": "I_0", "direction": 1}     # 반시계 방향
    ]
    st.session_state.points = [
        {"name": "p", "x": -1.0, "y": 0.0},
        {"name": "O", "x": 0.0, "y": 0.0},
        {"name": "q", "x": 1.0, "y": 0.0}
    ]
    st.session_state.symbol_values["I_0"] = 1.0

# -----------------------------------------------------------------------------
# 2. 물리 계산 엔진
# -----------------------------------------------------------------------------
def get_numeric_current(current_str, symbol_values):
    try:
        return float(current_str)
    except ValueError:
        return float(symbol_values.get(current_str, 1.0))

def calc_straight_wire_B(x, y, p1, p2, I, direction):
    x1, y1 = p1
    x2, y2 = p2
    dx = (x2 - x1) * direction
    dy = (y2 - y1) * direction
    line_len = np.hypot(dx, dy)
    
    if line_len < 1e-6:
        return np.zeros_like(x)

    cross_z = (dx * (y - y1) - dy * (x - x1))
    r = np.abs(cross_z) / line_len
    
    r_safe = np.where(r < 0.15, 1e-6, r)
    b_dir = np.sign(cross_z)
    
    B_z = (I / r_safe) * b_dir
    B_z[r < 0.15] = 0.0
    return B_z

# -----------------------------------------------------------------------------
# 3. 사이드바 UI
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ 도선 및 좌표 설정")

mode = st.sidebar.radio("작동 모드", ["설정 단계 (문제 구성)", "실행 단계 (자기장 중쇄/상쇄 시각화)"])
st.session_state.is_running = (mode == "실행 단계 (자기장 중쇄/상쇄 시각화)")

if st.sidebar.button("📚 첨부 예시 문제 불러오기"):
    load_preset_problem()
    st.rerun()

st.sidebar.markdown("---")

if not st.session_state.is_running:
    st.sidebar.subheader("➕ 도선 추가")
    w_type = st.sidebar.selectbox("도선 종류", ["직선 도선", "원형 도선 (반지름=d)"])
    w_name = st.sidebar.text_input("도선 이름", value=f"Wire_{len(st.session_state.wires)+1}")
    
    col1, col2 = st.sidebar.columns(2)
    if w_type == "직선 도선":
        with col1:
            x1 = st.number_input("시작 X", value=-2.0, step=1.0)
            y1 = st.number_input("시작 Y", value=-3.0, step=1.0)
        with col2:
            x2 = st.number_input("끝 X", value=-2.0, step=1.0)
            y2 = st.number_input("끝 Y", value=3.0, step=1.0)
    else:
        with col1:
            cx = st.number_input("중심 X", value=0.0, step=1.0)
        with col2:
            cy = st.number_input("중심 Y", value=-1.0, step=1.0)

    c_symbol = st.sidebar.text_input("전류 세기 (예: I_0, 2.0)", value="I_0")
    
    if st.sidebar.button("도선 설치"):
        if w_type == "직선 도선":
            st.session_state.wires.append({
                "type": "straight", "name": w_name,
                "p1": (x1, y1), "p2": (x2, y2),
                "current_symbol": c_symbol.strip(), "direction": 1
            })
        else:
            st.session_state.wires.append({
                "type": "circle", "name": w_name,
                "center": (cx, cy), "radius": 1.0,
                "current_symbol": c_symbol.strip(), "direction": 1
            })
        st.rerun()

# 도선 목록 제어
st.sidebar.subheader("📋 설치된 도선 목록")
for idx, wire in enumerate(st.session_state.wires):
    with st.sidebar.expander(f"도선 {wire['name']} ({'직선' if wire['type']=='straight' else '원형'})", expanded=False):
        new_sym = st.text_input(f"전류 기호 #{idx+1}", value=wire['current_symbol'], key=f"sym_{idx}")
        wire['current_symbol'] = new_sym.strip()
        
        col_dir, col_del = st.columns(2)
        with col_dir:
            if st.button(f"방향 반전 🔄", key=f"dir_{idx}"):
                wire['direction'] *= -1
                st.rerun()
        with col_del:
            if not st.session_state.is_running:
                if st.button("삭제 🗑️", key=f"del_{idx}"):
                    st.session_state.wires.pop(idx)
                    st.rerun()

# 미지수 전류 슬라이더
symbols = {w['current_symbol'] for w in st.session_state.wires if not w['current_symbol'].replace('.','',1).isdigit()}
if symbols:
    st.sidebar.subheader("🎛️ 미지수 전류 값 제어")
    for sym in sorted(list(symbols)):
        if sym not in st.session_state.symbol_values:
            st.session_state.symbol_values[sym] = 1.0
        val = st.sidebar.slider(f"미지수 [{sym}] 세기", -5.0, 5.0, float(st.session_state.symbol_values[sym]), 0.1, key=f"slider_{sym}")
        st.session_state.symbol_values[sym] = val

# -----------------------------------------------------------------------------
# 4. Plotly 좌표평면 시각화 (교재 물리 문제 스타일)
# -----------------------------------------------------------------------------
st.title("🧲 2D 도선 자기장 플랫폼")

grid_range = np.linspace(-3.5, 3.5, 141)
X, Y = np.meshgrid(grid_range, grid_range)
Z_total = np.zeros_like(X)

if st.session_state.is_running:
    for wire in st.session_state.wires:
        I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
        if wire['type'] == 'straight':
            Z_total += calc_straight_wire_B(X, Y, wire['p1'], wire['p2'], I_val, wire['direction'])

fig = go.Figure()

# 1) 실행 모드: 자기장 등고선/히트맵 overlay
if st.session_state.is_running and len(st.session_state.wires) > 0:
    Z_clipped = np.clip(Z_total, -8, 8)
    fig.add_trace(go.Contour(
        x=grid_range, y=grid_range, z=Z_clipped,
        colorscale='RdBu_r', zmin=-4, zmax=4, opacity=0.6,
        ncontours=25, showscale=True,
        colorbar=dict(title="자기장 B", tickvals=[-3, 0, 3], ticktext=["⊗ 들어감", "0 상쇄", "⊙ 나옴"])
    ))

# 2) 도선 및 라벨/화살표 그리기 (교재 스타일)
for wire in st.session_state.wires:
    sym = wire['current_symbol']
    name = wire['name']
    
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        if wire['direction'] == -1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        
        # 무한 도선처럼 보이도록 연장
        px1, py1 = x1 - ux * 10, y1 - uy * 10
        px2, py2 = x2 + ux * 10, y2 + uy * 10
        
        # 도선 본체 (이중선 느낌의 두꺼운 실선)
        fig.add_trace(go.Scatter(
            x=[px1, px2], y=[py1, py2],
            mode='lines', line=dict(color='#222222', width=3.5),
            showlegend=False, hoverinfo='none'
        ))
        
        # 전류 방향 화살표 및 수식 표기
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # 도선 명칭 (A, B, C 등)
        fig.add_annotation(
            x=x2 + uy * 0.2, y=y2 - ux * 0.2,
            text=f"<b>{name}</b>", showarrow=False,
            font=dict(size=16, color="black")
        )
        # 전류 화살표 및 전류값 (I_0)
        fig.add_annotation(
            x=mx + ux * 0.6, y=my + uy * 0.6, ax=mx, ay=my,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"  <b><i>{sym}</i></b>", font=dict(size=14, color="black"),
            align="left"
        )

    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        r = wire['radius']
        theta = np.linspace(0, 2*np.pi, 100)
        
        # 원형 도선
        fig.add_trace(go.Scatter(
            x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
            mode='lines', line=dict(color='#222222', width=2.5, dash='dash' if wire['direction']==-1 else 'solid'),
            showlegend=False, hoverinfo='none'
        ))
        
        # 원형 도선 명칭 및 중앙 전류 화살표
        fig.add_annotation(
            x=cx - r - 0.25, y=cy, text=f"<b>{name}</b>",
            showarrow=False, font=dict(size=16, color="black")
        )
        
        # 중심부 방향 화살표 (원 상단 기준 접선 방향)
        arrow_dir = 1 if wire['direction'] == 1 else -1
        fig.add_annotation(
            x=cx - 0.3*arrow_dir, y=cy + r, ax=cx + 0.3*arrow_dir, ay=cy + r,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"<b><i>{sym}</i></b>", font=dict(size=14, color="black")
        )

# 3) 주요 관찰 지점 (p, O, q 등 점 표시)
for pt in st.session_state.points:
    fig.add_trace(go.Scatter(
        x=[pt['x']], y=[pt['y']], mode='markers+text',
        marker=dict(size=7, color='black'),
        text=[f"<b>{pt['name']}</b>"], textposition="top center",
        font=dict(size=14, color="black"), showlegend=False
    ))

# 좌표축 및 레이아웃 설정 (교재 문제 스타일)
fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        range=[-3.2, 3.2], zeroline=True, zerolinecolor='black', zerolinewidth=1.5,
        tickvals=[-2, -1, 0, 1, 2], ticktext=["-2d", "-d", "O", "d", "2d"],
        gridcolor='#eeeeee', title="x"
    ),
    yaxis=dict(
        range=[-3.2, 3.2], zeroline=True, zerolinecolor='black', zerolinewidth=1.5,
        tickvals=[-2, -1, 0, 1, 2], ticktext=["-2d", "-d", "O", "d", "2d"],
        gridcolor='#eeeeee', title="y", scaleanchor="x", scaleratio=1
    ),
    width=680, height=680, margin=dict(l=30, r=30, t=30, b=30)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 지점별 자기장 식 및 계산값 출력
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 특정 지점의 자기장 수식 계산")

col_p1, col_p2 = st.columns(2)
with col_p1:
    target_x = st.number_input("X 위치 (-2d ~ 2d)", value=0.0, step=1.0)
with col_p2:
    target_y = st.number_input("Y 위치 (-2d ~ 2d)", value=0.0, step=1.0)

terms = []
num_total = 0.0

for wire in st.session_state.wires:
    sym = wire['current_symbol']
    I_val = get_numeric_current(sym, st.session_state.symbol_values)
    
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        dx, dy = (x2 - x1) * wire['direction'], (y2 - y1) * wire['direction']
        line_len = np.hypot(dx, dy)
        cross_z = (dx * (target_y - y1) - dy * (target_x - x1))
        r = np.abs(cross_z) / line_len
        
        if r > 0.05:
            sign = "+" if cross_z >= 0 else "-"
            r_str = f"{r:.0f}d" if r == int(r) else f"{r:.1f}d"
            terms.append(f"{sign} \\frac{{{sym}}}{{{r_str}}}")
            num_total += (I_val / r) * np.sign(cross_z)
            
    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        if abs(target_x - cx) < 1e-3 and abs(target_y - cy) < 1e-3:
            sign = "+" if wire['direction'] == 1 else "-"
            terms.append(f"{sign} \\frac{{{sym}}}{{d}} \\text{{ (원형 중심)}}")
            num_total += (I_val / 1.0) * wire['direction']

if terms:
    eq_str = "B_{total} = " + " ".join(terms)
    st.latex(eq_str)
    
    dir_desc = "지면을 뚫고 나오는 방향 (⊙)" if num_total > 0 else "지면을 뚫고 들어가는 방향 (⊗)" if num_total < 0 else "자기장 상쇄 (0)"
    st.info(f"**슬라이더 대입 결과:** $B = {abs(num_total):.2f} B_0$ | **방향:** {dir_desc}")
else:
    st.caption("해당 지점에 작용하는 유효한 자기장 항목이 없습니다.")
