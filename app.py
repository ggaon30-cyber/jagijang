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
    st.session_state.tool_mode = "straight"

if "p1_temp" not in st.session_state:
    st.session_state.p1_temp = None

if "last_processed_pt" not in st.session_state:
    st.session_state.last_processed_pt = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "target_coord" not in st.session_state:
    st.session_state.target_coord = (0.0, 0.0)

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
# 2. 물리 및 기하 계산 엔진
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

def get_perpendicular_foot_and_radius(target_pt, wire):
    """분석 지점에서 도선으로 내린 수선의 발 좌표와 반지름 계산"""
    tx, ty = target_pt
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx**2 + dy**2
        if len_sq < 1e-8:
            return (x1, y1), np.hypot(tx - x1, ty - y1)
        t = ((tx - x1) * dx + (ty - y1) * dy) / len_sq
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        radius = np.hypot(tx - foot_x, ty - foot_y)
        return (foot_x, foot_y), radius
    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        radius = np.hypot(tx - cx, ty - cy)
        return (cx, cy), radius

def calc_total_B_scalar(px, py, wires, symbol_values):
    """특정 (px, py) 위치에서의 합성 자기장 세기 계산 (상쇄 여부 측정용)"""
    total_b = 0.0
    for w in wires:
        I_val = get_numeric_current(w['current_symbol'], symbol_values)
        if w['type'] == 'straight':
            x1, y1 = w['p1']
            x2, y2 = w['p2']
            dx = (x2 - x1) * w['direction']
            dy = (y2 - y1) * w['direction']
            line_len = np.hypot(dx, dy)
            if line_len < 1e-6: continue
            cross_z = (dx * (py - y1) - dy * (px - x1))
            r = np.abs(cross_z) / line_len
            if r < 0.05: continue
            total_b += (I_val / r) * np.sign(cross_z)
        elif w['type'] == 'circle':
            cx, cy = w['center']
            k = w.get('b_scale', 1.0)
            if np.hypot(px - cx, py - cy) < 0.1:
                total_b += (I_val / 0.5) * k * w['direction']
    return total_b

# -----------------------------------------------------------------------------
# 3. 메인 인터페이스 & 도선/관찰지점 툴바
# -----------------------------------------------------------------------------
st.title("🧲 2D 도선 자기장 시각화 플랫폼")

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
        st.info("📍 **[관찰 지점 모드]** 좌표를 클릭하면 해당 위치에 관찰 지점이 즉시 생성됩니다.")
    else:
        st.caption("👆 [클릭 비활성화 모드] 화면 조작 시 요소가 추가되지 않습니다.")
else:
    st.info("📊 **[자기장 해석 모드]** 임의 지점을 클릭하면 해당 지점을 지나는 각 도선의 자기장 원형 궤적과 흐르는 입자(상쇄 구현)가 표시됩니다.")

# -----------------------------------------------------------------------------
# 4. 좌표평면 시각화 (Plotly)
# -----------------------------------------------------------------------------
fig = go.Figure()

# 1) 전체 자기장 등고선 (-20d ~ 20d)
if st.session_state.is_running and len(st.session_state.wires) > 0:
    grid_range = np.linspace(-20.0, 20.0, 200)
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
        colorscale='RdBu_r', zmin=-4, zmax=4, opacity=0.45,
        ncontours=25, showscale=True,
        colorbar=dict(title="자기장 B", tickvals=[-3, 0, 3], ticktext=["⊗ 들어감", "0 상쇄", "⊙ 나옴"])
    ))

# 2) 클릭용 배경 격자점
grid_x, grid_y = np.meshgrid(range(-20, 21), range(-20, 21))
fig.add_trace(go.Scatter(
    x=grid_x.flatten(), y=grid_y.flatten(),
    mode='markers', marker=dict(size=10, color='rgba(0,0,0,0.01)'),
    hoverinfo='x+y', showlegend=False
))

# 3) 도선 그리기
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

# 4) 설치된 관찰 지점
for pt in st.session_state.points:
    fig.add_trace(go.Scatter(
        x=[pt['x']], y=[pt['y']], mode='markers+text',
        marker=dict(size=9, color='#0044cc'),
        text=[f"<b>{pt['name']}</b>"], textposition="top center",
        textfont=dict(size=14, color="#0044cc"), showlegend=False
    ))

# -----------------------------------------------------------------------------
# 5. 자기장 해석 모드: 수선의 발 궤적 원 & 흘러가는 입자(상쇄 반영) 애니메이션
# -----------------------------------------------------------------------------
num_frames = 20
particles_per_circle = 24

if st.session_state.is_running and len(st.session_state.wires) > 0:
    target_pt = st.session_state.target_coord
    
    # 5-1) 각 도선별 궤적 원(실선 원) 그리기
    circle_info = []
    for w_idx, wire in enumerate(st.session_state.wires):
        foot, r_val = get_perpendicular_foot_and_radius(target_pt, wire)
        if r_val > 0.05:
            circle_info.append({
                "wire": wire,
                "foot": foot,
                "radius": r_val
            })
            # 원형 궤적 가이드 선
            theta_line = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=foot[0] + r_val * np.cos(theta_line),
                y=foot[1] + r_val * np.sin(theta_line),
                mode='lines',
                line=dict(color='rgba(230, 230, 230, 0.7)', width=1.5, dash='dot'),
                hoverinfo='none', showlegend=False
            ))

    # 5-2) 프레임 기반 흐르는 입자 위치 & 상쇄 반영 opacity 계산
    def generate_particle_data(frame_step):
        px_list, py_list, color_list, size_list = [], [], [], []
        rot_frac = frame_step / num_frames
        
        for c in circle_info:
            foot_x, foot_y = c["foot"]
            r_val = c["radius"]
            wire = c["wire"]
            rot_dir = wire.get('direction', 1)
            
            # 기준 초기각 (target_pt 관통하도록 설정)
            base_angle = np.arctan2(target_pt[1] - foot_y, target_pt[0] - foot_x)
            
            for p_i in range(particles_per_circle):
                angle = base_angle + (2 * np.pi * p_i / particles_per_circle) + (rot_dir * 2 * np.pi * rot_frac)
                px = foot_x + r_val * np.cos(angle)
                py = foot_y + r_val * np.sin(angle)
                
                # 위치별 합성 자기장 세기 계산 (상쇄 효과 반영)
                b_val = calc_total_B_scalar(px, py, st.session_state.wires, st.session_state.symbol_values)
                b_mag = abs(b_val)
                
                # 자기장이 상쇄되는 지점(b_mag ~ 0)에서는 투명해지고 약해짐
                alpha = np.clip(b_mag / 2.0, 0.05, 0.95)
                p_size = np.clip(4.0 + b_mag * 2.5, 2.0, 9.0)
                
                px_list.append(px)
                py_list.append(py)
                color_list.append(f"rgba(240, 240, 245, {alpha:.2f})")
                size_list.append(p_size)
                
        return px_list, py_list, color_list, size_list

    # Initial frame particle trace (Frame 0)
    init_px, init_py, init_colors, init_sizes = generate_particle_data(0)
    fig.add_trace(go.Scatter(
        x=init_px, y=init_py,
        mode='markers',
        marker=dict(size=init_sizes, color=init_colors, line=dict(color='rgba(100, 100, 100, 0.3)', width=0.5)),
        name="자기장 유체 입자", showlegend=False, hoverinfo='none'
    ))

    # Plotly Frame 생성 (애니메이션 구동용)
    particle_trace_idx = len(fig.data) - 1
    frames = []
    for f_idx in range(num_frames):
        fx, fy, fcols, fsizes = generate_particle_data(f_idx)
        frames.append(go.Frame(
            data=[go.Scatter(
                x=fx, y=fy,
                mode='markers',
                marker=dict(size=fsizes, color=fcols, line=dict(color='rgba(100, 100, 100, 0.3)', width=0.5))
            )],
            traces=[particle_trace_idx],
            name=f"frame_{f_idx}"
        ))
    fig.frames = frames

    # 애니메이션 버튼 컨트롤 설정
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            x=0.01, y=-0.05,
            xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="▶ 자기장 흐름 재생",
                    method="animate",
                    args=[None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True, "mode": "immediate", "loop": True}]
                ),
                dict(
                    label="⏸️ 일시정지",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                )
            ]
        )]
    )

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
# 6. 좌표 클릭 이벤트 처리 (오차 범위 Snap & 즉시 반영)
# -----------------------------------------------------------------------------
if selected_data and "selection" in selected_data and "points" in selected_data["selection"]:
    pts = selected_data["selection"]["points"]
    if len(pts) > 0:
        raw_x = float(pts[0]["x"])
        raw_y = float(pts[0]["y"])

        matched_coord = None
        for pt in st.session_state.points:
            if np.hypot(raw_x - pt['x'], raw_y - pt['y']) <= 0.4:
                matched_coord = (float(pt['x']), float(pt['y']))
                break

        if matched_coord is None:
            near_x, near_y = float(round(raw_x)), float(round(raw_y))
            if np.hypot(raw_x - near_x, raw_y - near_y) <= 0.45:
                matched_coord = (near_x, near_y)

        if matched_coord is not None:
            if st.session_state.is_running:
                if st.session_state.target_coord != matched_coord:
                    st.session_state.target_coord = matched_coord
                    st.rerun()
            else:
                if matched_coord != st.session_state.last_processed_pt:
                    st.session_state.last_processed_pt = matched_coord

                    if st.session_state.tool_mode == "straight":
                        if st.session_state.p1_temp is None:
                            st.session_state.p1_temp = matched_coord
                            st.rerun()
                        else:
                            p1, p2 = st.session_state.p1_temp, matched_coord
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
                            "type": "circle", "name": w_name, "center": matched_coord, "radius": 0.5,
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
                            "name": pt_name, "x": matched_coord[0], "y": matched_coord[1]
                        })
                        st.rerun()

# -----------------------------------------------------------------------------
# 7. 사이드바 제어판
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ 설치 요소 관리")

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

if st.session_state.wires:
    st.sidebar.subheader("📋 설치된 도선 목록")
    for idx, wire in enumerate(st.session_state.wires):
        col_w1, col_w2, col_w3, col_w4 = st.sidebar.columns([1.2, 1.4, 1.1, 0.8])
        
        with col_w1:
            st.write(f"**도선 {wire['name']}**")
            st.caption("직선" if wire['type']=='straight' else "원형")
            
        with col_w2:
            wire['current_symbol'] = st.text_input(
                "전류", value=wire['current_symbol'], key=f"sym_{idx}", label_visibility="collapsed"
            ).strip()
            
        with col_w3:
            dir_label = "⬆️" if wire['direction'] == 1 else "⬇️"
            if st.button(f"{dir_label} 반전", key=f"dir_{idx}", use_container_width=True):
                wire['direction'] *= -1
                st.rerun()
                
        with col_w4:
            if st.button("🗑️", key=f"del_{idx}", use_container_width=True):
                st.session_state.wires.pop(idx)
                st.session_state.p1_temp = None
                st.rerun()

        if wire['type'] == 'circle':
            wire['b_scale'] = st.sidebar.number_input(
                f"  └ 도선 {wire['name']} 중심 계수(k)", value=float(wire.get('b_scale', 1.0)), step=0.1, key=f"bscale_{idx}"
            )

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
# 8. 해석 모드 수식 및 계산 결과
# -----------------------------------------------------------------------------
if st.session_state.is_running:
    st.markdown("---")
    st.subheader("📐 선택 지점 자기장 계산")
    
    target_x, target_y = st.session_state.target_coord
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        if st.session_state.points:
            pt_options = [f"{pt['name']} ({pt['x']}d, {pt['y']}d)" for pt in st.session_state.points]
            
            matched_idx = 0
            for p_i, pt in enumerate(st.session_state.points):
                if (pt['x'], pt['y']) == (target_x, target_y):
                    matched_idx = p_i + 1
                    break

            selected_pt_str = st.selectbox(
                "🎯 빠른 선택 (클릭 설치된 관찰 지점)",
                ["선택 안 함"] + pt_options,
                index=matched_idx,
                key="target_selectbox"
            )
            
            if selected_pt_str != "선택 안 함":
                idx = pt_options.index(selected_pt_str)
                sel_x, sel_y = st.session_state.points[idx]['x'], st.session_state.points[idx]['y']
                if (sel_x, sel_y) != st.session_state.target_coord:
                    st.session_state.target_coord = (sel_x, sel_y)
                    st.rerun()

    with col_info:
        st.success(f"📍 현재 계산 위치: **X = {target_x}d, Y = {target_y}d** *(좌표평면 클릭 시 즉시 변경 및 입자 궤적 갱신)*")

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
