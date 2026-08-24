import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
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
    tx, ty = target_pt
    if wire['type'] == 'straight':
        (x1, y1), (x2, y2) = wire['p1'], wire['p2']
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx**2 + dy**2
        if len_sq < 1e-8:
            return (x1, y1), float(np.hypot(tx - x1, ty - y1))
        t = ((tx - x1) * dx + (ty - y1) * dy) / len_sq
        foot_x = float(x1 + t * dx)
        foot_y = float(y1 + t * dy)
        radius = float(np.hypot(tx - foot_x, ty - foot_y))
        return (foot_x, foot_y), radius
    elif wire['type'] == 'circle':
        cx, cy = wire['center']
        radius = float(np.hypot(tx - cx, ty - cy))
        return (float(cx), float(cy)), radius

# -----------------------------------------------------------------------------
# 3. 메인 인터페이스 & 툴바
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
    st.info("📊 **[자기장 해석 모드]** ")

# -----------------------------------------------------------------------------
# 4. 시각화 엔진
# -----------------------------------------------------------------------------
if not st.session_state.is_running:
    fig = go.Figure()

    grid_x, grid_y = np.meshgrid(range(-20, 21), range(-20, 21))
    fig.add_trace(go.Scatter(
        x=grid_x.flatten(), y=grid_y.flatten(),
        mode='markers', marker=dict(size=10, color='rgba(0,0,0,0.01)'),
        hoverinfo='x+y', showlegend=False
    ))

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

    for pt in st.session_state.points:
        fig.add_trace(go.Scatter(
            x=[pt['x']], y=[pt['y']], mode='markers+text',
            marker=dict(size=9, color='#0044cc'),
            text=[f"<b>{pt['name']}</b>"], textposition="top center",
            textfont=dict(size=14, color="#0044cc"), showlegend=False
        ))

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
        xaxis=dict(range=[-5.5, 5.5], zeroline=True, zerolinecolor='#444444', zerolinewidth=1.8, showgrid=True, gridcolor='#e5e5e5', tickvals=tick_range, ticktext=tick_labels, title="x"),
        yaxis=dict(range=[-5.5, 5.5], zeroline=True, zerolinecolor='#444444', zerolinewidth=1.8, showgrid=True, gridcolor='#e5e5e5', tickvals=tick_range, ticktext=tick_labels, title="y", scaleanchor="x", scaleratio=1),
        width=720, height=720, margin=dict(l=30, r=30, t=30, b=30)
    )

    selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="interactive_grid")

else:
    # 📊 [자기장 해석 모드]
    grid_range = np.linspace(-20.0, 20.0, 150)
    X, Y = np.meshgrid(grid_range, grid_range)
    Z_total = np.zeros_like(X)
    
    for wire in st.session_state.wires:
        I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
        if wire['type'] == 'straight':
            Z_total += calc_straight_wire_B(X, Y, wire['p1'], wire['p2'], I_val, wire['direction'])
        elif wire['type'] == 'circle':
            k_val = wire.get('b_scale', 1.0)
            Z_total += calc_circle_wire_B(X, Y, wire['center'], wire.get('radius', 0.5), I_val, wire['direction'], k_scale=k_val)

    Z_clipped = np.clip(Z_total, -8, 8).tolist()

    straight_info_list = []
    circle_info_list = []
    target_pt = st.session_state.target_coord

    for wire in st.session_state.wires:
        foot, r_val = get_perpendicular_foot_and_radius(target_pt, wire)
        if wire['type'] == 'straight' and r_val > 0.05:
            I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
            b_mag = abs(I_val / r_val)

            (x1, y1), (x2, y2) = wire['p1'], wire['p2']
            dx = (x2 - x1) * wire['direction']
            dy = (y2 - y1) * wire['direction']
            cross_z = dx * (target_pt[1] - y1) - dy * (target_pt[0] - x1)
            b_sign = 1 if cross_z >= 0 else -1
            
            straight_info_list.append({
                "foot": [float(foot[0]), float(foot[1])],
                "radius": float(r_val),
                "bMag": float(b_mag),
                "direction": int(wire.get('direction', 1)),
                "p1": [float(wire['p1'][0]), float(wire['p1'][1])],
                "p2": [float(wire['p2'][0]), float(wire['p2'][1])]
            })
        elif wire['type'] == 'circle':
            I_val = get_numeric_current(wire['current_symbol'], st.session_state.symbol_values)
            circle_info_list.append({
                "center": [float(wire['center'][0]), float(wire['center'][1])],
                "radius": float(wire.get('radius', 0.5)),
                "bMag": float(abs(I_val / 0.5)),
                "direction": int(wire.get('direction', 1))
            })

    wires_json = json.dumps(st.session_state.wires)
    points_json = json.dumps(st.session_state.points)
    symbols_json = json.dumps(st.session_state.symbol_values)
    target_json = json.dumps(st.session_state.target_coord)
    straights_json = json.dumps(straight_info_list)
    circles_json = json.dumps(circle_info_list)
    grid_x_json = json.dumps(grid_range.tolist())
    z_json = json.dumps(Z_clipped)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            #container {{ position: relative; width: 720px; height: 720px; }}
            #plotly_canvas {{ position: absolute; top: 0; left: 0; width: 720px; height: 720px; }}
            #particle_canvas {{ position: absolute; top: 0; left: 0; width: 720px; height: 720px; pointer-events: none; z-index: 10; }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="plotly_canvas"></div>
            <canvas id="particle_canvas" width="720" height="720"></canvas>
        </div>
        <script>
            const wires = {wires_json};
            const points = {points_json};
            const symbols = {symbols_json};
            const targetPt = {target_json};
            const straightCircles = {straights_json};
            const circleWires = {circles_json};
            const gridX = {grid_x_json};
            const zData = {z_json};

            function getI(sym) {{
                let val = parseFloat(sym);
                return isNaN(val) ? (symbols[sym] !== undefined ? symbols[sym] : 1.0) : val;
            }}

            function calcTotalB(px, py) {{
                let totalB = 0;
                for (let w of wires) {{
                    let I = getI(w.current_symbol);
                    if (w.type === 'straight') {{
                        let x1 = w.p1[0], y1 = w.p1[1];
                        let x2 = w.p2[0], y2 = w.p2[1];
                        let dx = (x2 - x1) * w.direction;
                        let dy = (y2 - y1) * w.direction;
                        let lineLen = Math.hypot(dx, dy);
                        if (lineLen < 1e-6) continue;
                        let crossZ = dx * (py - y1) - dy * (px - x1);
                        let r = Math.abs(crossZ) / lineLen;
                        if (r < 0.05) continue;
                        totalB += (I / r) * Math.sign(crossZ);
                    }} else if (w.type === 'circle') {{
                        let cx = w.center[0], cy = w.center[1];
                        let k = w.b_scale || 1.0;
                        if (Math.hypot(px - cx, py - cy) < 0.1) {{
                            totalB += (I / 0.5) * k * w.direction;
                        }}
                    }}
                }}
                return totalB;
            }}

            let traces = [];

            // 1) Contour
            traces.push({{
                x: gridX, y: gridX, z: zData,
                type: 'contour', colorscale: 'RdBu_r', zmin: -4, zmax: 4, opacity: 0.35,
                ncontours: 25, showscale: true,
                colorbar: {{ title: '자기장 B', tickvals: [-3, 0, 3], ticktext: ['⊗ 들어감', '0 상쇄', '⊙ 나옴'] }}
            }});

            // 2) 직선 도선용 관찰 지점 통과 궤적 점선 (kParallel 축소로 이심률 향상)
            const kParallel = 0.35; // 기존 0.50 -> 0.35로 이심률 증가 (더 납작한 타원)
            const kPerp = 1.00;

            for (let c of straightCircles) {{
                let ux = 1, uy = 0, nx = 0, ny = 1;
                let dx = (c.p2[0] - c.p1[0]) * c.direction;
                let dy = (c.p2[1] - c.p1[1]) * c.direction;
                let len = Math.hypot(dx, dy);
                if (len > 1e-5) {{ ux = dx / len; uy = dy / len; nx = -uy; ny = ux; }}

                let gx = [], gy = [];
                for (let a = 0; a <= 2*Math.PI; a += 0.04) {{
                    let xOff = c.radius * kParallel * Math.cos(a) * ux + c.radius * kPerp * Math.sin(a) * nx;
                    let yOff = c.radius * kParallel * Math.cos(a) * uy + c.radius * kPerp * Math.sin(a) * ny;
                    gx.push(c.foot[0] + xOff);
                    gy.push(c.foot[1] + yOff);
                }}
                traces.push({{
                    x: gx, y: gy, mode: 'lines',
                    line: {{ color: 'rgba(140, 140, 155, 0.55)', width: 1.1, dash: 'dot' }},
                    hoverinfo: 'none', showlegend: false
                }});
            }}

            // 3) Wires
            for (let w of wires) {{
                if (w.type === 'straight') {{
                    let x1 = w.p1[0], y1 = w.p1[1], x2 = w.p2[0], y2 = w.p2[1];
                    if (w.direction === -1) {{ [x1, y1, x2, y2] = [x2, y2, x1, y1]; }}
                    let dx = x2 - x1, dy = y2 - y1;
                    let len = Math.hypot(dx, dy);
                    if (len > 1e-5) {{
                        let ux = dx / len, uy = dy / len;
                        traces.push({{
                            x: [x1 - ux*40, x2 + ux*40], y: [y1 - uy*40, y2 + uy*40],
                            mode: 'lines', line: {{ color: '#111111', width: 3.8 }}, showlegend: false
                        }});
                    }}
                }} else if (w.type === 'circle') {{
                    let cx = w.center[0], cy = w.center[1], r = w.radius || 0.5;
                    let wx = [], wy = [];
                    for (let a = 0; a <= 2*Math.PI; a += 0.06) {{
                        wx.push(cx + r * Math.cos(a));
                        wy.push(cy + r * Math.sin(a));
                    }}
                    traces.push({{
                        x: wx, y: wy, mode: 'lines',
                        line: {{ color: '#111111', width: 3, dash: w.direction === -1 ? 'dash' : 'solid' }}, showlegend: false
                    }});
                }}
            }}

            // 4) Points
            for (let pt of points) {{
                traces.push({{
                    x: [pt.x], y: [pt.y], mode: 'markers+text',
                    marker: {{ size: 9, color: '#0044cc' }},
                    text: ['<b>' + pt.name + '</b>'], textposition: 'top center',
                    textfont: {{ size: 14, color: '#0044cc' }}, showlegend: false
                }});
            }}

            // 5) Target Point
            traces.push({{
                x: [targetPt[0]], y: [targetPt[1]], mode: 'markers',
                marker: {{ size: 14, color: 'green', symbol: 'cross' }}, showlegend: false
            }});

            let tickRange = [], tickLabels = [];
            for (let i = -20; i <= 20; i++) {{
                tickRange.push(i);
                tickLabels.push(i === 0 ? "O" : i + "d");
            }}

            let layout = {{
                template: 'plotly_white',
                xaxis: {{ range: [-5.5, 5.5], zeroline: true, zerolinecolor: '#444444', zerolinewidth: 1.8, tickvals: tickRange, ticktext: tickLabels, title: 'x' }},
                yaxis: {{ range: [-5.5, 5.5], zeroline: true, zerolinecolor: '#444444', zerolinewidth: 1.8, tickvals: tickRange, ticktext: tickLabels, title: 'y', scaleanchor: 'x', scaleratio: 1 }},
                width: 720, height: 720, margin: {{ l: 30, r: 30, t: 30, b: 30 }}
            }};

            Plotly.newPlot('plotly_canvas', traces, layout);

            // Canvas 애니메이션 엔진
            const pCanvas = document.getElementById('particle_canvas');
            const ctx = pCanvas.getContext('2d');
            const gd = document.getElementById('plotly_canvas');

            // 원형 도선 광선
            let circleRays = [];
            for (let c of circleWires) {{
                let rayArray = [];
                let numRays = 36;
                for (let k = 0; k < numRays; k++) {{
                    let lenFactor = 0.25 + Math.random() * 0.15;
                    rayArray.push({{
                        angle: Math.random() * 2 * Math.PI,
                        progress: Math.random() * (1.0 + lenFactor),
                        speed: 0.035 + Math.random() * 0.025,
                        lenFactor: lenFactor
                    }});
                }}
                circleRays.push({{ circle: c, rays: rayArray }});
            }}

            let frameStep = 0;

            function animateParticles() {{
                ctx.clearRect(0, 0, 720, 720);

                if (!gd._fullLayout || !gd._fullLayout.xaxis) {{
                    requestAnimationFrame(animateParticles);
                    return;
                }}

                let xaxis = gd._fullLayout.xaxis;
                let yaxis = gd._fullLayout.yaxis;

                // -------------------------------------------------------------
                // 엔진 (1) 직선 도선 자기장: 세기 및 회전 속도 축소 조정
                // -------------------------------------------------------------
                for (let c of straightCircles) {{
                    let footX = c.foot[0], footY = c.foot[1];
                    let rBase = c.radius;
                    let bMag = c.bMag;
                    let rotDir = c.direction;

                    let dx = (c.p2[0] - c.p1[0]) * c.direction;
                    let dy = (c.p2[1] - c.p1[1]) * c.direction;
                    let len = Math.hypot(dx, dy);
                    let ux = 1, uy = 0, nx = 0, ny = 1;
                    if (len > 1e-5) {{ ux = dx / len; uy = dy / len; nx = -uy; ny = ux; }}

                    // 회전 속도 감쇄 (기존 0.7 배율 -> 0.5 배율)
                    let speedMult = Math.min(Math.max(0.5 + 0.6 * bMag, 0.5), 2.5) * 0.7;
                    let rOffsets = [0.0];
                    let count = 16;

                    if (bMag >= 1.5) {{ rOffsets = [-0.03, 0.0, 0.03]; count = 24; }}
                    else if (bMag >= 0.7) {{ rOffsets = [-0.025, 0.025]; count = 20; }}

                    let baseAngle = Math.atan2(targetPt[1] - footY, targetPt[0] - footX);
                    let rotFrac = (frameStep % 200) / 200.0;

                    for (let rOff of rOffsets) {{
                        let rCurr = rBase + rOff;
                        for (let i = 0; i < count; i++) {{
                            let angle = baseAngle + (2 * Math.PI * i / count) + (rotDir * 2 * Math.PI * rotFrac * speedMult);
                            let cosA = Math.cos(angle);
                            let sinA = Math.sin(angle);

                            let px = footX + (rCurr * kParallel * cosA) * ux + (rCurr * kPerp * sinA) * nx;
                            let py = footY + (rCurr * kParallel * cosA) * uy + (rCurr * kPerp * sinA) * ny;

                            let bNet = calcTotalB(px, py);
                            // 진하기 약 -15% 축소 조정
                            let alpha = Math.min((Math.abs(bNet) / 1.5) * 0.85, 0.80);
                            if (alpha < 0.05) continue;

                            let screenX = xaxis.l2p(px) + xaxis._offset;
                            let screenY = yaxis.l2p(py) + yaxis._offset;

                            let vx = (-rCurr * kParallel * sinA * ux + rCurr * kPerp * cosA * nx) * rotDir;
                            let vy = (-rCurr * kParallel * sinA * uy + rCurr * kPerp * cosA * ny) * rotDir;

                            let screenVx = vx * (xaxis._length / (xaxis.range[1] - xaxis.range[0]));
                            let screenVy = -vy * (yaxis._length / (yaxis.range[1] - yaxis.range[0]));
                            let tangentAngle = Math.atan2(screenVy, screenVx);

                            let grayVal = bMag >= 1.2 ? 30 : (bMag >= 0.6 ? 65 : 110);
                            let colorStr = `rgba(${{grayVal}}, ${{grayVal + 5}}, ${{grayVal + 10}}, ${{alpha.toFixed(2)}})`;

                            ctx.save();
                            ctx.translate(screenX, screenY);
                            ctx.rotate(tangentAngle);
                            ctx.beginPath();
                            ctx.ellipse(0, 0, 8.0, 1.8, 0, 0, 2 * Math.PI);
                            ctx.fillStyle = colorStr;
                            ctx.fill();
                            ctx.strokeStyle = `rgba(0, 0, 0, ${{Math.min(alpha + 0.1, 0.85)}})`;
                            ctx.lineWidth = 0.8;
                            ctx.stroke();
                            ctx.restore();
                        }}
                    }}
                }}

                // -------------------------------------------------------------
                // 엔진 (2) 원형 도선 중심 광선 연출
                // -------------------------------------------------------------
                for (let cr of circleRays) {{
                    let c = cr.circle;
                    let cx = c.center[0], cy = c.center[1];
                    let rBound = c.radius || 0.5;
                    let isOutwards = (c.direction === 1);

                    for (let r of cr.rays) {{
                        r.progress += r.speed;

                        if (r.progress >= 1.0 + r.lenFactor) {{
                            r.progress = 0.0;
                            r.angle = Math.random() * 2 * Math.PI;
                            r.speed = 0.035 + Math.random() * 0.025;
                            r.lenFactor = 0.25 + Math.random() * 0.15;
                        }}

                        let pHead, pTail;
                        if (isOutwards) {{
                            pHead = Math.min(1.0, r.progress);
                            pTail = Math.max(0.0, r.progress - r.lenFactor);
                        }} else {{
                            pHead = Math.max(0.0, 1.0 - r.progress);
                            pTail = Math.min(1.0, 1.0 - (r.progress - r.lenFactor));
                        }}

                        if (Math.abs(pHead - pTail) > 0.001) {{
                            let rHead = pHead * (rBound * 0.95);
                            let rTail = pTail * (rBound * 0.95);

                            let hx = cx + rHead * Math.cos(r.angle);
                            let hy = cy + rHead * Math.sin(r.angle);
                            let tx = cx + rTail * Math.cos(r.angle);
                            let ty = cy + rTail * Math.sin(r.angle);

                            let screenHx = xaxis.l2p(hx) + xaxis._offset;
                            let screenHy = yaxis.l2p(hy) + yaxis._offset;
                            let screenTx = xaxis.l2p(tx) + xaxis._offset;
                            let screenTy = yaxis.l2p(ty) + yaxis._offset;

                            ctx.save();
                            ctx.beginPath();
                            ctx.moveTo(screenTx, screenTy);
                            ctx.lineTo(screenHx, screenHy);
                            ctx.strokeStyle = "rgba(0, 0, 0, 0.95)";
                            ctx.lineWidth = 0.55;
                            ctx.lineCap = "round";
                            ctx.stroke();
                            ctx.restore();
                        }}
                    }}
                }}

                frameStep++;
                requestAnimationFrame(animateParticles);
            }}

            requestAnimationFrame(animateParticles);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=730, width=730)

# -----------------------------------------------------------------------------
# 5. 좌표 클릭 이벤트 처리 (편집 모드 시)
# -----------------------------------------------------------------------------
if not st.session_state.is_running and 'selected_data' in locals():
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

            if matched_coord is not None and matched_coord != st.session_state.last_processed_pt:
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
# 6. 사이드바 제어판
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
# 7. 수식 및 대입 결과
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
                "🎯 해석 타겟 지점 선택",
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
        st.success(f"📍 현재 계산 위치: **X = {target_x}d, Y = {target_y}d**")

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
