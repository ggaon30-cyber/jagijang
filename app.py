import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 및 세션 상태 초기화
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

if "tool_mode" not in st.session_state:
    st.session_state.tool_mode = "straight"  # "straight", "circle", "select"

if "first_click" not in st.session_state:
    st.session_state.first_click = None

if "last_processed_pt" not in st.session_state:
    st.session_state.last_processed_pt = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# 예시 문제 불러오기
def load_preset_problem():
    st.session_state.wires = [
        {"type": "straight", "name": "A", "p1": (-2.0, -5.0), "p2": (-2.0, 5.0), "current_symbol": "I_0", "direction": -1},
        {"type": "straight", "name": "B", "p1": (2.0, -5.0), "p2": (2.0, 5.0), "current_symbol": "I_0", "direction": 1},
        {"type": "straight", "name": "C", "p1": (-5.0, -2.0), "p2": (5.0, -2.0), "current_symbol": "I_0", "direction": 1},
        {"type": "circle", "name": "D", "center": (0.0, -1.0), "radius": 1.0, "current_symbol": "I_0", "direction": 1}
    ]
    st.session_state.points = [
        {"name": "p", "x": -1.0, "y": 0.0},
        {"name": "O", "x": 0.0, "y": 0.0},
        {"name": "q", "x": 1.0, "y": 0.0}
    ]
    st.session_state.symbol_values["I_0"] = 1.0
    st.session_state.first_click = None

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
# 3. 메인 인터페이스 & 도선 그리기 툴바
# -----------------------------------------------------------------------------
st.title("🧲 2D 도선 자기장 플랫폼")

col_mode, col_preset = st.columns([3, 1])
with col_mode:
    mode_selection = st.radio(
        "작동 모드 선택",
        ["🔨 도선 그리기 모드", "📊 자기장 해석 모드"],
        horizontal=True
    )
    st.session_state.is_running = (mode_selection == "📊 자기장 해석 모드")

with col_preset:
    if st.button("📚 교재 예시 문제 불러오기", use_container_width=True):
        load_preset_problem()
        st.rerun()

st.markdown("---")

# 도선 설치 도구 선택창
if not st.session_state.is_running:
    col_t1, col_t2, col_t3, col_t4 = st.columns([1, 1, 1, 1])
    with col_t1:
        if st.button("📏 직선 도선 (두 점 클릭)", use_container_width=True, type="primary" if st.session_state.tool_mode == "straight" else "secondary"):
            st.session_state.tool_mode = "straight"
            st.session_state.first_click = None
            st.rerun()
    with col_t2:
        if st.button("⭕ 원형 도선 (한 점 클릭)", use_container_width=True, type="primary" if st.session_state.tool_mode == "circle" else "secondary"):
            st.session_state.tool_mode = "circle"
            st.session_state.first_click = None
            st.rerun()
    with col_t3:
        if st.button("👆 클릭 비활성화", use_container_width=True, type="primary" if st.session_state.tool_mode == "select" else "secondary"):
            st.session_state.tool_mode = "select"
            st.session_state.first_click = None
            st.rerun()
    with col_t4:
        if st.button("🧹 전체 도선 삭제", use_container_width=True):
            st.session_state.wires = []
            st.session_state.first_click = None
            st.rerun()

    # 안내 메시지
    if st.session_state.tool_mode == "straight":
        if st.session_state.first_click is None:
            st.info("📍 **[직선 도선 모드]** 좌표평면 위에서 **첫 번째 점**을 클릭하세요.")
        else:
            fc = st.session_state.first_click
            st.warning(f"📍 **첫 번째 점 선택 완료 ({fc[0]}d, {fc[1]}d)** → 직선을 완성할 **두 번째 점**을 클릭하세요.")
            if st.button("첫 번째 점 선택 취소"):
                st.session_state.first_click = None
                st.rerun()
    elif st.session_state.tool_mode == "circle":
        st.info("⭕ **[원형 도선 모드]** 좌표평면 위에서 **중심이 될 점**을 클릭하면 반지름 1d인 원형 도선이 설치됩니다.")
    else:
        st.caption("👆 [클릭 비활성화 모드] 클릭으로 도선이 추가되지 않습니다.")

# -----------------------------------------------------------------------------
# 4. 균일한 5d 좌표평면 그리드 시각화
# -----------------------------------------------------------------------------
fig = go.Figure()

# 1) 해석 모드 시 자기장 등고선 표시
if st.session_state.is_running and len(st.session_state.wires) > 0:
    grid_range = np.linspace(-5.5, 5.5, 180)
    X, Y = np.meshgrid(grid_range, grid_range)
    Z_total = np.zeros_like(X)
    
    for wire in st.session_state.wires:
        I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
        if wire['type'] == 'straight':
            Z_total += calc_straight_wire_B(X, Y, wire['p1'], wire['p2'], I_val, wire['direction'])
            
    Z_clipped = np.clip(Z_total, -8, 8)
    fig.add_trace(go.Contour(
        x=grid_range, y=grid_range, z=Z_clipped,
        colorscale='RdBu_r', zmin=-4, zmax=4, opacity=0.55,
        ncontours=25, showscale=True,
        colorbar=dict(title="자기장 B", tickvals=[-3, 0, 3], ticktext=["⊗ 들어감", "0 상쇄", "⊙ 나옴"])
    ))

# 2) 클릭 가능한 배경 격자 포인트 (격자점 클릭 인식용)
grid_x, grid_y = np.meshgrid(range(-5, 6), range(-5, 6))
fig.add_trace(go.Scatter(
    x=grid_x.flatten(), y=grid_y.flatten(),
    mode='markers',
    marker=dict(size=12, color='rgba(0,0,0,0.01)'),
    hoverinfo='x+y',
    showlegend=False
))

# 3) 도선 및 방향 화살표 그리기
for wire in st.session_state.wires:
    sym = wire['current_symbol']
    name = wire['name']
    
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        if wire['direction'] == -1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-5:
            continue
        ux, uy = dx / length, dy / length
        
        # 무한 직선으로 연장
        px1, py1 = x1 - ux * 20, y1 - uy * 20
        px2, py2 = x2 + ux * 20, y2 + uy * 20
        
        fig.add_trace(go.Scatter(
            x=[px1, px2], y=[py1, py2],
            mode='lines', line=dict(color='#111111', width=3),
            showlegend=False, hoverinfo='none'
        ))
        
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        fig.add_annotation(
            x=x2 + uy * 0.25, y=y2 - ux * 0.25,
            text=f"<b>{name}</b>", showarrow=False,
            font=dict(size=15, color="black")
        )
        fig.add_annotation(
            x=mx + ux * 0.5, y=my + uy * 0.5, ax=mx, ay=my,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"  <b><i>{sym}</i></b>", font=dict(size=13, color="black"),
            align="left"
        )

    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        r = wire['radius']
        theta = np.linspace(0, 2*np.pi, 100)
        
        fig.add_trace(go.Scatter(
            x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
            mode='lines', line=dict(color='#111111', width=2.5, dash='dash' if wire['direction']==-1 else 'solid'),
            showlegend=False, hoverinfo='none'
        ))
        
        fig.add_annotation(
            x=cx - r - 0.3, y=cy, text=f"<b>{name}</b>",
            showarrow=False, font=dict(size=15, color="black")
        )
        
        arrow_dir = 1 if wire['direction'] == 1 else -1
        fig.add_annotation(
            x=cx - 0.3*arrow_dir, y=cy + r, ax=cx + 0.3*arrow_dir, ay=cy + r,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"<b><i>{sym}</i></b>", font=dict(size=13, color="black")
        )

# 4) 첫 번째 클릭 지점 하이라이트 표시
if st.session_state.first_click is not None:
    fc = st.session_state.first_click
    fig.add_trace(go.Scatter(
        x=[fc[0]], y=[fc[1]], mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        showlegend=False
    ))

# 5) 관찰 지점 (p, O, q)
for pt in st.session_state.points:
    fig.add_trace(go.Scatter(
        x=[pt['x']], y=[pt['y']], mode='markers+text',
        marker=dict(size=8, color='black'),
        text=[f"<b>{pt['name']}</b>"], textposition="top center",
        textfont=dict(size=14, color="black"), showlegend=False
    ))

# 균일한 5d 간격 좌표평면 그리드 설정
tick_range = list(range(-5, 6))
tick_labels = ["-5d", "-4d", "-3d", "-2d", "-d", "O", "d", "2d", "3d", "4d", "5d"]

fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        range=[-5.5, 5.5],
        zeroline=True, zerolinecolor='black', zerolinewidth=2,
        showgrid=True, gridcolor='#e0e0e0', gridwidth=1,
        tickvals=tick_range, ticktext=tick_labels,
        title="x"
    ),
    yaxis=dict(
        range=[-5.5, 5.5],
        zeroline=True, zerolinecolor='black', zerolinewidth=2,
        showgrid=True, gridcolor='#e0e0e0', gridwidth=1,
        tickvals=tick_range, ticktext=tick_labels,
        title="y", scaleanchor="x", scaleratio=1
    ),
    width=720, height=720,
    margin=dict(l=30, r=30, t=30, b=30)
)

# Plotly 차트 출력 및 클릭 이벤트 감지
selected_data = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="interactive_grid"
)

# -----------------------------------------------------------------------------
# 5. 클릭 좌표 이벤트 처리 (도선 자동 생성)
# -----------------------------------------------------------------------------
if selected_data and "selection" in selected_data and "points" in selected_data["selection"]:
    pts = selected_data["selection"]["points"]
    if len(pts) > 0 and not st.session_state.is_running:
        clicked_x = float(round(pts[0]["x"]))
        clicked_y = float(round(pts[0]["y"]))
        curr_pt = (clicked_x, clicked_y)

        if curr_pt != st.session_state.last_processed_pt:
            st.session_state.last_processed_pt = curr_pt

            if st.session_state.tool_mode == "straight":
                if st.session_state.first_click is None:
                    st.session_state.first_click = curr_pt
                    st.rerun()
                else:
                    p1 = st.session_state.first_click
                    p2 = curr_pt
                    if p1 != p2:
                        w_name = chr(65 + len(st.session_state.wires))
                        st.session_state.wires.append({
                            "type": "straight", "name": w_name,
                            "p1": p1, "p2": p2,
                            "current_symbol": "I_0", "direction": 1
                        })
                    st.session_state.first_click = None
                    st.rerun()

            elif st.session_state.tool_mode == "circle":
                w_name = chr(65 + len(st.session_state.wires))
                st.session_state.wires.append({
                    "type": "circle", "name": w_name,
                    "center": curr_pt, "radius": 1.0,
                    "current_symbol": "I_0", "direction": 1
                })
                st.rerun()

# -----------------------------------------------------------------------------
# 6. 하단 사이드바 및 도선 데이터 속성 제어
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ 도선 상세 제어")

if st.session_state.wires:
    st.sidebar.subheader("📋 설치된 도선 목록")
    for idx, wire in enumerate(st.session_state.wires):
        with st.sidebar.expander(f"도선 {wire['name']} ({'직선' if wire['type']=='straight' else '원형'})", expanded=False):
            wire['current_symbol'] = st.text_input(f"전류 기호 #{idx+1}", value=wire['current_symbol'], key=f"sym_{idx}").strip()
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("방향 반전 🔄", key=f"dir_{idx}"):
                    wire['direction'] *= -1
                    st.rerun()
            with c2:
                if st.button("삭제 🗑️", key=f"del_{idx}"):
                    st.session_state.wires.pop(idx)
                    st.session_state.first_click = None
                    st.rerun()

symbols = {w['current_symbol'] for w in st.session_state.wires if not w['current_symbol'].replace('.','',1).isdigit()}
if symbols:
    st.sidebar.subheader("🎛️ 미지수 전류 값 슬라이더")
    for sym in sorted(list(symbols)):
        if sym not in st.session_state.symbol_values:
            st.session_state.symbol_values[sym] = 1.0
        st.session_state.symbol_values[sym] = st.sidebar.slider(
            f"미지수 [{sym}] 세기", -5.0, 5.0, float(st.session_state.symbol_values[sym]), 0.1, key=f"slider_{sym}"
        )

# 수식 계산 결과 표시
if st.session_state.is_running:
    st.markdown("---")
    st.subheader("📐 지정 위치 자기장 수식 계산")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        target_x = st.number_input("X 위치 (-5d ~ 5d)", value=0.0, step=1.0)
    with col_p2:
        target_y = st.number_input("Y 위치 (-5d ~ 5d)", value=0.0, step=1.0)

    terms = []
    num_total = 0.0
    for wire in st.session_state.wires:
        sym = wire['current_symbol']
        I_val = get_numeric_current(sym, st.session_state.symbol_values)
        
        if wire['type'] == 'straight':
            (x1, y1), (x2, y2) = wire['p1'], wire['p2']
            dx, dy = (x2 - x1) * wire['direction'], (y2 - y1) * wire['direction']
            line_len = np.hypot(dx, dy)
            if line_len < 1e-5: continue
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
        st.latex("B_{total} = " + " ".join(terms))
        dir_desc = "지면을 뚫고 나오는 방향 (⊙)" if num_total > 0 else "지면을 뚫고 들어가는 방향 (⊗)" if num_total < 0 else "자기장 상쇄 (0)"
        st.info(f"**대입 결과:** $B = {abs(num_total):.2f} B_0$ | **방향:** {dir_desc}")
