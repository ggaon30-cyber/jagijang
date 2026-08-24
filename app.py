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
    st.session_state.tool_mode = "straight"  # "straight", "circle", "point", "select"

if "p1_temp" not in st.session_state:
    st.session_state.p1_temp = None

if "last_processed_pt" not in st.session_state:
    st.session_state.last_processed_pt = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "target_coord" not in st.session_state:
    st.session_state.target_coord = (0.0, 0.0)

# 예시 문제 불러오기
def load_preset_problem():
    st.session_state.wires = [
        {"type": "straight", "name": "A", "p1": (-2.0, -20.0), "p2": (-2.0, 20.0), "current_symbol": "I_A", "direction": -1},
        {"type": "straight", "name": "B", "p1": (2.0, -20.0), "p2": (2.0, 20.0), "current_symbol": "I_B", "direction": 1},
        {"type": "straight", "name": "C", "p1": (-20.0, -2.0), "p2": (20.0, -2.0), "current_symbol": "I_C", "direction": 1},
        {"type": "circle", "name": "D", "center": (0.0, -1.0), "radius": 0.5, "current_symbol": "I_D", "direction": 1, "b_scale": 1.0}
    ]
    st.session_state.points = [
        {"name": "p", "x": -1.0, "y": 0.0},
        {"name": "O", "x": 0.0, "y": 0.0},
        {"name": "q", "x": 1.0, "y": 0.0}
    ]
    st.session_state.symbol_values = {
        "I_A": 1.0,
        "I_B": 1.0,
        "I_C": 1.0,
        "I_D": 1.0
    }
    st.session_state.p1_temp = None

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
    r_safe = np.where(r < 0.1, 1e-6, r)
    b_dir = np.sign(cross_z)
    
    B_z = (I / r_safe) * b_dir
    B_z[r < 0.1] = 0.0
    return B_z

def calc_circle_wire_B(x, y, center, radius, I, direction, k_scale=1.0, num_segments=32):
    cx, cy = center
    dtheta = 2 * np.pi / num_segments
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    Bz = np.zeros_like(x)
    
    for a in angles:
        wx = cx + radius * np.cos(a)
        wy = cy + radius * np.sin(a)
        dlx = -radius * np.sin(a) * dtheta * direction
        dly =  radius * np.cos(a) * dtheta * direction
        
        rx = x - wx
        ry = y - wy
        dist_sq = rx**2 + ry**2 + 0.02
        dist = np.sqrt(dist_sq)
        
        Bz += (dlx * ry - dly * rx) / (dist**3)
        
    norm_factor = (k_scale * I * radius) / (2 * np.pi)
    return Bz * norm_factor

# -----------------------------------------------------------------------------
# 3. 메인 인터페이스 & 도선/관찰지점 툴바
# -----------------------------------------------------------------------------
st.title("🧲 2D 도선 자기장 플랫폼")

col_mode, col_preset = st.columns([3, 1])
with col_mode:
    mode_selection = st.radio(
        "작동 모드 선택",
        ["🔨 도선/관찰지점 편집 모드", "📊 자기장 해석 모드"],
        horizontal=True
    )
    st.session_state.is_running = (mode_selection == "📊 자기장 해석 모드")

with col_preset:
    if st.button("📚 교재 예시 문제 불러오기", use_container_width=True):
        load_preset_problem()
        st.rerun()

st.markdown("---")

if not st.session_state.is_running:
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([1.2, 1.2, 1.2, 1, 1])
    with col_t1:
        if st.button("📏 직선 도선 (2점 클릭)", use_container_width=True, type="primary" if st.session_state.tool_mode == "straight" else "secondary"):
            st.session_state.tool_mode = "straight"
            st.session_state.p1_temp = None
            st.rerun()
    with col_t2:
        if st.button("⭕ 원형 도선 (반지름 0.5d)", use_container_width=True, type="primary" if st.session_state.tool_mode == "circle" else "secondary"):
            st.session_state.tool_mode = "circle"
            st.session_state.p1_temp = None
            st.rerun()
    with col_t3:
        if st.button("📍 관찰 지점 클릭 설치", use_container_width=True, type="primary" if st.session_state.tool_mode == "point" else "secondary"):
            st.session_state.tool_mode = "point"
            st.session_state.p1_temp = None
            st.rerun()
    with col_t4:
        if st.button("👆 클릭 비활성화", use_container_width=True, type="primary" if st.session_state.tool_mode == "select" else "secondary"):
            st.session_state.tool_mode = "select"
            st.session_state.p1_temp = None
            st.rerun()
    with col_t5:
        if st.button("🧹 전체 요소 삭제", use_container_width=True):
            st.session_state.wires = []
            st.session_state.points = []
            st.session_state.p1_temp = None
            st.rerun()

    if st.session_state.tool_mode == "straight":
        if st.session_state.p1_temp is None:
            st.info("📏 **[직선 도선 모드]** 첫 번째 점을 클릭하세요.")
        else:
            p1 = st.session_state.p1_temp
            st.warning(f"📍 **첫 번째 점 선택됨 ({p1[0]}d, {p1[1]}d)** → 두 번째 점을 클릭하면 직선이 생성됩니다.")
    elif st.session_state.tool_mode == "circle":
        st.info("⭕ **[원형 도선 모드]** 좌표를 클릭하면 중심 반지름 0.5d 원형 도선이 설치됩니다.")
    elif st.session_state.tool_mode == "point":
        st.info("📍 **[관찰 지점 모드]** 좌표를 클릭하면 해당 위치에 관찰 지점(p, O, q...)이 즉시 생성됩니다.")
    else:
        st.caption("👆 [클릭 비활성화 모드] 화면 조작 시 요소가 추가되지 않습니다.")
else:
    st.info("📊 **[자기장 해석 모드]** 좌표평면의 임의 지점이나 설치된 관찰 지점을 클릭하면 해당 위치의 합성 자기장이 즉시 계산됩니다.")

# -----------------------------------------------------------------------------
# 4. 좌표평면 시각화
# -----------------------------------------------------------------------------
fig = go.Figure()

# 1) 자기장 등고선 (해석 모드)
if st.session_state.is_running and len(st.session_state.wires) > 0:
    grid_range = np.linspace(-6.0, 6.0, 160)
    X, Y = np.meshgrid(grid_range, grid_range)
    Z_total = np.zeros_like(X)
    
    for wire in st.session_state.wires:
        I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
        if wire['type'] == 'straight':
            Z_total += calc_straight_wire_B(X, Y, wire['p1'], wire['p2'], I_val, wire['direction'])
        elif wire['type'] == 'circle':
            k_val = wire.get('b_scale', 1.0)
            Z_total += calc_circle_wire_B(X, Y, wire['center'], wire.get('radius', 0.5), I_val, wire['direction'], k_scale=k_val)
            
    Z_clipped = np.clip(Z_total, -8, 8)
    fig.add_trace(go.Contour(
        x=grid_range, y=grid_range, z=Z_clipped,
        colorscale='RdBu_r', zmin=-4, zmax=4, opacity=0.55,
        ncontours=25, showscale=True,
        colorbar=dict(title="자기장 B", tickvals=[-3, 0, 3], ticktext=["⊗ 들어감", "0 상쇄", "⊙ 나옴"])
    ))

# 2) 광범위 격자점 (-20d ~ 20d)
grid_x, grid_y = np.meshgrid(range(-20, 21), range(-20, 21))
fig.add_trace(go.Scatter(
    x=grid_x.flatten(), y=grid_y.flatten(),
    mode='markers',
    marker=dict(size=10, color='rgba(0,0,0,0.01)'),
    hoverinfo='x+y', showlegend=False
))

# 3) 도선 그려내기
for wire in st.session_state.wires:
    sym, name = wire['current_symbol'], wire['name']
    
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        if wire['direction'] == -1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-5: continue
        ux, uy = dx / length, dy / length
        
        fig.add_trace(go.Scatter(
            x=[x1 - ux * 40, x2 + ux * 40], y=[y1 - uy * 40, y2 + uy * 40],
            mode='lines', line=dict(color='#111111', width=3.8),
            showlegend=False, hoverinfo='none'
        ))
        
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        fig.add_annotation(x=x2 + uy * 0.25, y=y2 - ux * 0.25, text=f"<b>{name}</b>", showarrow=False, font=dict(size=15, color="black"))
        fig.add_annotation(
            x=mx + ux * 0.5, y=my + uy * 0.5, ax=mx, ay=my, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"  <b><i>{sym}</i></b>", font=dict(size=13, color="black"), align="left"
        )

    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        r = wire.get('radius', 0.5)
        theta = np.linspace(0, 2*np.pi, 100)
        
        fig.add_trace(go.Scatter(
            x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
            mode='lines', line=dict(color='#111111', width=3.0, dash='dash' if wire['direction']==-1 else 'solid'),
            showlegend=False, hoverinfo='none'
        ))
        fig.add_annotation(x=cx - r - 0.25, y=cy, text=f"<b>{name}</b>", showarrow=False, font=dict(size=15, color="black"))
        arrow_dir = 1 if wire['direction'] == 1 else -1
        fig.add_annotation(
            x=cx - 0.2*arrow_dir, y=cy + r, ax=cx + 0.2*arrow_dir, ay=cy + r, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="black",
            text=f"<b><i>{sym}</i></b>", font=dict(size=13, color="black")
        )

# 4) 클릭으로 설치된 관찰 지점들 표시
for pt in st.session_state.points:
    fig.add_trace(go.Scatter(
        x=[pt['x']], y=[pt['y']], mode='markers+text',
        marker=dict(size=9, color='#0044cc'),
        text=[f"<b>{pt['name']}</b>"], textposition="top center",
        textfont=dict(size=14, color="#0044cc"), showlegend=False
    ))

# 현재 선택된 해석 계산 대상점 표시 (초록 십자)
if st.session_state.is_running:
    tx, ty = st.session_state.target_coord
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty], mode='markers',
        marker=dict(size=14, color='green', symbol='cross'),
        showlegend=False, hoverinfo='none'
    ))

# 클릭 진행 중 첫 점 표시
if st.session_state.p1_temp is not None:
    p1 = st.session_state.p1_temp
    fig.add_trace(go.Scatter(
        x=[p1[0]], y=[p1[1]], mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        showlegend=False
    ))

tick_range = list(range(-20, 21))
tick_labels = [f"{i}d" if i != 0 else "O" for i in range(-20, 21)]

fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        range=[-5.5, 5.5], zeroline=True, zerolinecolor='#444444', zerolinewidth=1.8,
        showgrid=True, gridcolor='#e5e5e5', gridwidth=0.8, tickvals=tick_range, ticktext=tick_labels, title="x"
    ),
    yaxis=dict(
        range=[-5.5, 5.5], zeroline=True, zerolinecolor='#444444', zerolinewidth=1.8,
        showgrid=True, gridcolor='#e5e5e5', gridwidth=0.8, tickvals=tick_range, ticktext=tick_labels, title="y",
        scaleanchor="x", scaleratio=1
    ),
    width=720, height=720, margin=dict(l=30, r=30, t=30, b=30)
)

selected_data = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", selection_mode="points", key="interactive_grid"
)

# -----------------------------------------------------------------------------
# 5. 좌표 클릭 이벤트 처리 (관찰지점 / 도선 / 해석 타겟 설정)
# -----------------------------------------------------------------------------
if selected_data and "selection" in selected_data and "points" in selected_data["selection"]:
    pts = selected_data["selection"]["points"]
    if len(pts) > 0:
        clicked_x = float(round(pts[0]["x"]))
        clicked_y = float(round(pts[0]["y"]))
        curr_pt = (clicked_x, clicked_y)

        if curr_pt != st.session_state.last_processed_pt:
            st.session_state.last_processed_pt = curr_pt

            # 해석 모드일 때 클릭한 위치를 계산 대상 위치로 변경
            if st.session_state.is_running:
                st.session_state.target_coord = curr_pt
                st.rerun()

            # 편집 모드일 때 선택 도구별 설치
            else:
                if st.session_state.tool_mode == "straight":
                    if st.session_state.p1_temp is None:
                        st.session_state.p1_temp = curr_pt
                        st.rerun()
                    else:
                        p1, p2 = st.session_state.p1_temp, curr_pt
                        if p1 != p2:
                            w_name = chr(65 + len(st.session_state.wires))
                            curr_symbol = f"I_{w_name}"
                            st.session_state.wires.append({
                                "type": "straight", "name": w_name, "p1": p1, "p2": p2,
                                "current_symbol": curr_symbol, "direction": 1
                            })
                            if curr_symbol not in st.session_state.symbol_values:
                                st.session_state.symbol_values[curr_symbol] = 1.0
                        st.session_state.p1_temp = None
                        st.rerun()

                elif st.session_state.tool_mode == "circle":
                    w_name = chr(65 + len(st.session_state.wires))
                    curr_symbol = f"I_{w_name}"
                    st.session_state.wires.append({
                        "type": "circle", "name": w_name, "center": curr_pt, "radius": 0.5,
                        "current_symbol": curr_symbol, "direction": 1, "b_scale": 1.0
                    })
                    if curr_symbol not in st.session_state.symbol_values:
                        st.session_state.symbol_values[curr_symbol] = 1.0
                    st.rerun()

                elif st.session_state.tool_mode == "point":
                    default_names = ["p", "O", "q", "r", "s", "t", "u", "v"]
                    cnt = len(st.session_state.points)
                    pt_name = default_names[cnt] if cnt < len(default_names) else f"P{cnt+1}"
                    
                    st.session_state.points.append({
                        "name": pt_name, "x": clicked_x, "y": clicked_y
                    })
                    st.rerun()

# -----------------------------------------------------------------------------
# 6. 사이드바 제어판
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ 설치 요소 관리")

# 1) 클릭 설치된 관찰 지점 목록 & 관리
if st.session_state.points:
    st.sidebar.subheader("📍 클릭 설치된 관찰 지점")
    for p_idx, pt in enumerate(st.session_state.points):
        col_pname, col_pos, col_pdel = st.sidebar.columns([1.5, 1.5, 1])
        with col_pname:
            pt['name'] = st.text_input(f"이름", value=pt['name'], key=f"pt_name_{p_idx}", label_visibility="collapsed")
        with col_pos:
            st.caption(f"({pt['x']}d, {pt['y']}d)")
        with col_pdel:
            if st.button("삭제", key=f"pdel_{p_idx}"):
                st.session_state.points.pop(p_idx)
                st.rerun()

# 2) 도선 상세 제어
if st.session_state.wires:
    st.sidebar.subheader("📋 설치된 도선 목록")
    for idx, wire in enumerate(st.session_state.wires):
        with st.sidebar.expander(f"도선 {wire['name']} ({'직선' if wire['type']=='straight' else '원형'})", expanded=False):
            wire['current_symbol'] = st.text_input(f"전류 기호 #{idx+1}", value=wire['current_symbol'], key=f"sym_{idx}").strip()
            
            if wire['type'] == 'circle':
                wire['b_scale'] = st.number_input(
                    f"중심 자기장 세기 계수 (k)", value=float(wire.get('b_scale', 1.0)), step=0.1, key=f"bscale_{idx}"
                )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("방향 반전 🔄", key=f"dir_{idx}"):
                    wire['direction'] *= -1
                    st.rerun()
            with c2:
                if st.button("삭제 🗑️", key=f"del_{idx}"):
                    st.session_state.wires.pop(idx)
                    st.session_state.p1_temp = None
                    st.rerun()

# 3) 미지수 전류 슬라이더
symbols = {w['current_symbol'] for w in st.session_state.wires if not w['current_symbol'].replace('.','',1).isdigit()}
if symbols:
    st.sidebar.subheader("🎛️ 미지수 전류 값 슬라이더")
    for sym in sorted(list(symbols)):
        if sym not in st.session_state.symbol_values:
            st.session_state.symbol_values[sym] = 1.0
        st.session_state.symbol_values[sym] = st.sidebar.slider(
            f"미지수 [{sym}] 세기", -5.0, 5.0, float(st.session_state.symbol_values[sym]), 0.1, key=f"slider_{sym}"
        )

# -----------------------------------------------------------------------------
# 7. 해석 모드 수식 및 계산 결과
# -----------------------------------------------------------------------------
if st.session_state.is_running:
    st.markdown("---")
    st.subheader("📐 선택 지점 자기장 계산")
    
    target_x, target_y = st.session_state.target_coord
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        if st.session_state.points:
            pt_options = [f"{pt['name']} ({pt['x']}d, {pt['y']}d)" for pt in st.session_state.points]
            selected_pt_str = st.selectbox("🎯 빠른 선택 (클릭 설치된 관찰 지점)", ["선택 안 함"] + pt_options)
            if selected_pt_str != "선택 안 함":
                idx = pt_options.index(selected_pt_str)
                target_x = st.session_state.points[idx]['x']
                target_y = st.session_state.points[idx]['y']
                st.session_state.target_coord = (target_x, target_y)

    with col_info:
        st.success(f"📍 현재 계산 위치: **X = {target_x}d, Y = {target_y}d** *(좌표평면을 직접 클릭하면 바뀝니다)*")

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
            k = wire.get('b_scale', 1.0)
            if abs(target_x - cx) < 1e-3 and abs(target_y - cy) < 1e-3:
                sign = "+" if wire['direction'] == 1 else "-"
                k_str = f"{k:.1f} \\cdot " if k != 1.0 else ""
                terms.append(f"{sign} {k_str}\\frac{{{sym}}}{{0.5d}} \\text{{ (원형 중심)}}")
                num_total += (I_val / 0.5) * k * wire['direction']

    if terms:
        st.latex("B_{total} = " + " ".join(terms))
        dir_desc = "지면을 뚫고 나오는 방향 (⊙)" if num_total > 0 else "지면을 뚫고 들어가는 방향 (⊗)" if num_total < 0 else "자기장 상쇄 (0)"
        st.info(f"**대입 결과:** $B = {abs(num_total):.2f} B_0$ | **방향:** {dir_desc}")
    else:
        st.caption("선택한 지점에 유효하게 작용하는 자기장 수식이 없습니다.")
