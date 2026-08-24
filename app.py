import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 및 세션 상태 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="2D 도선 자기장 시각화 플랫폼", layout="wide")

if "wires" not in st.session_state:
    # 도선 목록: dict 형태로 저장
    # type: 'straight' | 'circle'
    # p1, p2: (x, y) coordinates for straight wire
    # center: (x, y) for circle wire
    # current_symbol: str ('1.0', '2.0', 'I_1', 'I_2', 'A', 'B' 등)
    # direction: 1 또는 -1 (화살표 방향 반전용)
    st.session_state.wires = []

if "symbol_values" not in st.session_state:
    # 미지수 전류의 슬라이더 값 저장 dict {symbol_name: float_value}
    st.session_state.symbol_values = {}

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# -----------------------------------------------------------------------------
# 2. 자기장 계산 물리 엔진
# -----------------------------------------------------------------------------
def get_numeric_current(current_str, symbol_values):
    """전류 입력을 수치값으로 변환 (숫자면 float, 미지수면 symbol_values에서 참조)"""
    try:
        return float(current_str)
    except ValueError:
        return float(symbol_values.get(current_str, 1.0))

def calc_straight_wire_B(x, y, p1, p2, I, direction):
    """
    무한 직선 도선에 의한 자기장 (Z축 방향 수직 성분 B_z)
    B = (I / r) * (오른손 법칙에 따른 방향)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # 도선의 방향 벡터 d = (dx, dy)
    dx = (x2 - x1) * direction
    dy = (y2 - y1) * direction
    line_len = np.hypot(dx, dy)
    
    if line_len < 1e-6:
        return np.zeros_like(x), np.ones_like(x, dtype=bool)

    # 외적을 이용한 수직 거리 r 계산
    # |(x2-x1)(y1-y) - (x1-x)(y2-y1)| / line_len
    cross_z = (dx * (y - y1) - dy * (x - x1))
    r = np.abs(cross_z) / line_len
    
    # 도선 자체에 너무 가까운 점은 마스킹 (특이점 방지)
    on_wire = r < 0.15
    r_safe = np.where(on_wire, 1e-6, r)
    
    # 오른손 법칙: (dx, dy, 0) x (px, py, 0) 의 z성분 부호가 자기장 방향(지면을 뚫고 나오는 방향 + / 들어가는 방향 -)
    b_dir = np.sign(cross_z)
    
    # 자기장 크기 B = I / r (상대적 단위)
    B_z = (I / r_safe) * b_dir
    B_z[on_wire] = 0.0
    
    return B_z, on_wire

# -----------------------------------------------------------------------------
# 3. 사이드바 UI - 도선 설치 및 제어판
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ 도선 설정 제어판")

mode = st.sidebar.radio("모드 선택", ["설정 단계 (도선 배치)", "실행 단계 (자기장 시각화)"])
st.session_state.is_running = (mode == "실행 단계 (자기장 시각화)")

if not st.session_state.is_running:
    st.sidebar.subheader("➕ 새 도선 추가")
    wire_type = st.sidebar.selectbox("도선 종류", ["직선 도선", "원형 도선 (반지름=1)"])
    
    col1, col2 = st.sidebar.columns(2)
    if wire_type == "직선 도선":
        with col1:
            x1 = st.number_input("시작점 X1", value=-3.0, step=1.0)
            y1 = st.number_input("시작점 Y1", value=-3.0, step=1.0)
        with col2:
            x2 = st.number_input("끝점 X2", value=3.0, step=1.0)
            y2 = st.number_input("끝점 Y2", value=3.0, step=1.0)
    else:
        with col1:
            cx = st.number_input("중심점 X", value=0.0, step=1.0)
        with col2:
            cy = st.number_input("중심점 Y", value=0.0, step=1.0)

    c_symbol = st.sidebar.text_input("전류 세기 (숫자 또는 미지수 이름 ex: 2.0, I1, A)", value="I1")
    
    if st.sidebar.button("도선 추가하기"):
        if wire_type == "직선 도선":
            if x1 == x2 and y1 == y2:
                st.sidebar.error("시작점과 끝점이 같을 수 없습니다.")
            else:
                st.session_state.wires.append({
                    "type": "straight",
                    "p1": (x1, y1),
                    "p2": (x2, y2),
                    "current_symbol": c_symbol.strip(),
                    "direction": 1
                })
        else:
            st.session_state.wires.append({
                "type": "circle",
                "center": (cx, cy),
                "radius": 1.0,
                "current_symbol": c_symbol.strip(),
                "direction": 1
            })
        st.rerun()

# 기존 도선 목록 및 제어
st.sidebar.subheader("📋 설치된 도선 목록")
if len(st.session_state.wires) == 0:
    st.sidebar.info("설치된 도선이 없습니다. 위에서 도선을 추가하세요.")
else:
    for idx, wire in enumerate(st.session_state.wires):
        with st.sidebar.expander(f"도선 #{idx+1} ({'직선' if wire['type'] == 'straight' else '원형'})", expanded=True):
            if wire['type'] == 'straight':
                st.write(f"위치: ({wire['p1'][0]}, {wire['p1'][1]}) ➔ ({wire['p2'][0]}, {wire['p2'][1]})")
            else:
                st.write(f"중심: ({wire['center'][0]}, {wire['center'][1]}), 반지름: 1")
            
            # 전류 세기/미지수 변경 (실행 단계에서도 변경 가능)
            new_sym = st.text_input(f"전류 설정 #{idx+1}", value=wire['current_symbol'], key=f"sym_{idx}")
            wire['current_symbol'] = new_sym.strip()
            
            col_dir, col_del = st.columns(2)
            with col_dir:
                dir_label = "방향: 正 ➔" if wire['direction'] == 1 else "방향: 逆 ↵"
                if st.button(f"방향 반전 ({dir_label})", key=f"dir_{idx}"):
                    wire['direction'] *= -1
                    st.rerun()
            with col_del:
                if not st.session_state.is_running:
                    if st.button("삭제 🗑️", key=f"del_{idx}"):
                        st.session_state.wires.pop(idx)
                        st.rerun()

# 미지수 슬라이더 자동 생성
symbols = set()
for w in st.session_state.wires:
    sym = w['current_symbol']
    try:
        float(sym)
    except ValueError:
        if sym:
            symbols.add(sym)

if symbols:
    st.sidebar.subheader("🎛️ 미지수 전류 조절 슬라이더")
    for sym in sorted(list(symbols)):
        if sym not in st.session_state.symbol_values:
            st.session_state.symbol_values[sym] = 1.0
        val = st.sidebar.slider(f"미지수 [{sym}] 값", min_value=-5.0, max_value=5.0, value=float(st.session_state.symbol_values[sym]), step=0.1, key=f"slider_{sym}")
        st.session_state.symbol_values[sym] = val

# -----------------------------------------------------------------------------
# 4. 시각화 시뮬레이션 영역 (Plotly 렌더링)
# -----------------------------------------------------------------------------
st.title("🧲 2D 도선 자기장 중쇄 및 상쇄 시각화")
st.caption("격자 평면 위에서 도선에 의한 자기장의 중쇄(강화) 및 상쇄 현상을 확인하세요. (지면을 뚫고 나오는 방향: Red / 들어가는 방향: Blue)")

# 2D 좌표격자 생성 (-6 ~ +6)
grid_range = np.linspace(-6, 6, 121)
X, Y = np.meshgrid(grid_range, grid_range)
Z_total = np.zeros_like(X)

# 자기장 합성 계산
if st.session_state.is_running:
    for wire in st.session_state.wires:
        I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
        if wire['type'] == 'straight':
            Bz, _ = calc_straight_wire_B(X, Y, wire['p1'], wire['p2'], I_val, wire['direction'])
            Z_total += Bz

fig = go.Figure()

# 1) 실행 단계일 때 자기장 히트맵 & 컨투어(유동적 흐름 표현) 추가
if st.session_state.is_running and len(st.session_state.wires) > 0:
    # 자기장 세기 등고선 (클리핑 처리로 시각적 가독성 확보)
    Z_clipped = np.clip(Z_total, -10, 10)
    
    fig.add_trace(go.Contour(
        x=grid_range,
        y=grid_range,
        z=Z_clipped,
        colorscale='RdBu_r', # 빨강(+z, 뚫고 나옴) / 파랑(-z, 뚫고 들어감) / 백색(0, 상쇄)
        zmin=-5,
        zmax=5,
        ncontours=30,
        line_width=0.8,
        colorbar=dict(title="자기장 세기 B (상대값)", tickvals=[-4, 0, 4], ticktext=["-B (들어감 ⊗)", "0 (상쇄)", "+B (나옴 ⊙)"]),
        hoverinfo='none'
    ))

# 2) 배경 격자 점 그리기
dot_x, dot_y = np.meshgrid(np.arange(-6, 7, 1), np.arange(-6, 7, 1))
dot_x = dot_x.flatten()
dot_y = dot_y.flatten()

# 원점 강조
is_origin = (dot_x == 0) & (dot_y == 0)
fig.add_trace(go.Scatter(
    x=dot_x[~is_origin], y=dot_y[~is_origin],
    mode='markers',
    marker=dict(size=3, color='gray', opacity=0.5),
    name='격자점',
    hoverinfo='none'
))
fig.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(size=8, color='black'),
    name='원점 (0,0)',
    hoverinfo='text',
    hovertext='원점 (0,0)'
))

# 3) 도선 그리기
for idx, wire in enumerate(st.session_state.wires):
    sym = wire['current_symbol']
    cur_val = get_numeric_current(sym, st.session_state.symbol_values)
    
    if wire['type'] == 'straight':
        x1, y1 = wire['p1']
        x2, y2 = wire['p2']
        
        # 방향 반영
        if wire['direction'] == -1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # 연장선 계산 (무한 직선 도선 시각화)
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        
        px1, py1 = x1 - ux * 15, y1 - uy * 15
        px2, py2 = x2 + ux * 15, y2 + uy * 15
        
        # 직선 그리기
        fig.add_trace(go.Scatter(
            x=[px1, px2], y=[py1, py2],
            mode='lines',
            line=dict(color='orange', width=4),
            name=f"직선도선 #{idx+1}",
            hoverinfo='none'
        ))
        
        # 전류 방향 화살표 표시
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        fig.add_annotation(
            x=mid_x + ux * 0.5, y=mid_y + uy * 0.5,
            ax=mid_x, ay=mid_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="red",
            text=f"<b>I={sym}</b>",
            font=dict(size=14, color="black"),
            bgcolor="yellow"
        )
        
    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx + np.cos(theta)
        circle_y = cy + np.sin(theta)
        
        # 원형 도선 테두리
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='green', width=4),
            name=f"원형도선 #{idx+1}",
            hoverinfo='none'
        ))
        
        # 원형 도선 중심 표시 및 방향 화살표
        dir_text = "↺ (CCW, +z)" if wire['direction'] == 1 else "↻ (CW, -z)"
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='text+markers',
            marker=dict(size=10, color='green', symbol='x'),
            text=[f"<b>중심<br>I={sym}<br>{dir_text}</b>"],
            textposition="top center",
            name=f"원형도선 중심 #{idx+1}"
        ))

# 레이아웃 설정
fig.update_layout(
    xaxis=dict(range=[-6, 6], zeroline=True, gridcolor='lightgray', dtick=1),
    yaxis=dict(range=[-6, 6], zeroline=True, gridcolor='lightgray', dtick=1, scaleanchor="x", scaleratio=1),
    width=750,
    height=750,
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=False)
# -----------------------------------------------------------------------------
# 5. 수식 및 해석 정보 영역 (마우스 탐색 / 좌표 선택)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 임의의 점 자기장 수식 및 대입 결과 확인")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    inspect_x = st.number_input("탐색할 X 좌표", value=0.0, step=0.5)
with col_sel2:
    inspect_y = st.number_input("탐색할 Y 좌표", value=1.0, step=0.5)

# 수식 생성 로직
terms_symbolic = []
numeric_sum = 0.0

for idx, wire in enumerate(st.session_state.wires):
    sym = wire['current_symbol']
    I_val = get_numeric_current(sym, st.session_state.symbol_values)
    
    if wire['type'] == 'straight':
        p1, p2 = wire['p1'], wire['p2']
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = (x2 - x1) * wire['direction'], (y2 - y1) * wire['direction']
        line_len = np.hypot(dx, dy)
        
        cross_z = (dx * (inspect_y - y1) - dy * (inspect_x - x1))
        r = np.abs(cross_z) / line_len
        
        if r < 0.05:
            terms_symbolic.append(f"[도선 #{idx+1} 정의불가(도선 위)]")
        else:
            sign = "+" if cross_z >= 0 else "-"
            terms_symbolic.append(f"{sign} \\frac{{{sym}}}{{{r:.2f}}}")
            b_val = (I_val / r) * np.sign(cross_z)
            numeric_sum += b_val
            
    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        # 원형 도선은 중심에서만 고려 (요구사항)
        if abs(inspect_x - cx) < 1e-4 and abs(inspect_y - cy) < 1e-4:
            sign = "+" if wire['direction'] == 1 else "-"
            terms_symbolic.append(f"{sign} {sym} \\text{{ (원형도선 중심)}}")
            numeric_sum += I_val * wire['direction']

# 수식 출력
if len(terms_symbolic) == 0:
    st.info("현재 위치에 작용하는 자기장이 없거나 도선이 설정되지 않았습니다.")
else:
    latex_eq = "B = " + " ".join(terms_symbolic)
    st.latex(latex_eq)
    
    # 1B, 2B와 같은 단위로 결과 표시
    unit_b = f"{numeric_sum:.2f} B_0"
    if numeric_sum > 0:
        dir_str = "지면을 뚫고 나오는 방향 (⊙)"
    elif numeric_sum < 0:
        dir_str = "지면을 뚫고 들어가는 방향 (⊗)"
    else:
        dir_str = "자기장 상쇄 (0)"
        
    st.success(f"**최종 계산 결과:** $B = {unit_b}$  |  **방향:** {dir_str}")
