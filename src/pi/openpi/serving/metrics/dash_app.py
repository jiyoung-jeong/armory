"""Dash-based metrics dashboard."""

from __future__ import annotations

import datetime

import dash
from dash import Input
from dash import Output
from dash import Patch
from dash import State
from dash import ctx
from dash import dcc
from dash import html
import numpy as np
from openpi_client.schemas import ServerMetadata
import plotly.graph_objects as go

from openpi.serving.metrics.store import MetricsStore
from openpi.serving.metrics.store import Snapshot

# ---------------------------------------------------------------------------
# Plotly dark theme helpers
# ---------------------------------------------------------------------------
_DARK: dict = {
    "paper_bgcolor": "#1a1a1a",
    "plot_bgcolor": "#111",
    "font": {"color": "#bbb", "family": "'SF Mono', monospace", "size": 11},
    "margin": {"t": 32, "r": 16, "b": 48, "l": 56},
    "xaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
    "yaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
    "legend": {"bgcolor": "#1a1a1a", "bordercolor": "#333"},
}


def _layout(**extra: object) -> dict:
    return {
        **_DARK,
        **extra,
        "xaxis": {**_DARK["xaxis"], **extra.get("xaxis", {})},  # type: ignore[arg-type]
        "yaxis": {**_DARK["yaxis"], **extra.get("yaxis", {})},  # type: ignore[arg-type]
    }


# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------
_BODY = {
    "background": "#111",
    "color": "#ccc",
    "fontFamily": "'SF Mono', 'Fira Code', monospace",
    "padding": "24px",
    "fontSize": "14px",
    "minHeight": "100vh",
}
_CARD = {
    "background": "#1a1a1a",
    "border": "1px solid #222",
    "borderRadius": "6px",
    "overflow": "hidden",
    "marginBottom": "12px",
}
_CARD_HDR = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "padding": "10px 14px",
    "borderBottom": "1px solid #222",
    "fontSize": "0.8em",
    "color": "#666",
}
_SECTION_TITLE = {
    "fontSize": "0.7em",
    "textTransform": "uppercase",
    "letterSpacing": "1.5px",
    "color": "#555",
    "borderBottom": "1px solid #1e1e1e",
    "paddingBottom": "5px",
    "marginBottom": "12px",
}
_STAT_CARD = {
    "background": "#1a1a1a",
    "border": "1px solid #222",
    "borderRadius": "6px",
    "padding": "14px 16px",
}
_INPUT = {
    "background": "#1e1e1e",
    "border": "1px solid #333",
    "color": "#ccc",
    "padding": "3px 8px",
    "borderRadius": "3px",
    "fontFamily": "inherit",
    "fontSize": "0.95em",
}
_BTN_PRIMARY = {
    "background": "#0d2d4a",
    "border": "1px solid #2979ff",
    "color": "#82b1ff",
    "padding": "6px 16px",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontFamily": "inherit",
    "fontSize": "0.85em",
}
_BTN_ACTIVE = {
    "background": "#14351f",
    "border": "1px solid #66bb6a",
    "color": "#b9f6ca",
    "padding": "6px 16px",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontFamily": "inherit",
    "fontSize": "0.85em",
}
_CFG = {"displaylogo": False, "responsive": True}


# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------
def _section(title: str, *children: object) -> html.Div:
    return html.Div(
        style={"marginBottom": "28px"},
        children=[html.Div(title.upper(), style=_SECTION_TITLE), *children],
    )


def _stat_card(value: str, label: str, *, value_color: str = "#4fc3f7") -> html.Div:
    return html.Div(
        style=_STAT_CARD,
        children=[
            html.Div(value, style={"fontSize": "1.9em", "fontWeight": 700, "color": value_color, "lineHeight": "1.1"}),
            html.Div(label, style={"color": "#555", "fontSize": "0.72em", "marginTop": "5px"}),
        ],
    )


def _robot_table(robots: dict, sla_pct: float) -> html.Element:
    if not robots:
        return html.Div("No robots yet.", style={"color": "#444", "padding": "12px"})
    th = {
        "textAlign": "left",
        "color": "#555",
        "fontWeight": 400,
        "padding": "6px 12px",
        "borderBottom": "1px solid #222",
    }
    td_base = {"padding": "6px 12px"}
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85em"},
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Robot", style=th),
                        html.Th("Starved / Obs", style=th),
                        html.Th("Starvation (%)", style=th),
                        html.Th("Healthy", style=th),
                        html.Th("TP (suc/sec/robot)", style=th),
                        html.Th("Avg Net Delay (ms)", style=th),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(rid, style={**td_base, "color": "#ccc"}),
                            html.Td(
                                f"{r.get('starved_steps', 0):,} / {r.get('observed_steps', 0):,}",
                                style={
                                    **td_base,
                                    "color": "#ff8a65" if r.get("starved_steps", 0) > 0 else "#4fc3f7",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                f"{r.get('starvation_rate_pct', 0.0):.2f}",
                                style={
                                    **td_base,
                                    "color": "#ff8a65" if r.get("starvation_rate_pct", 0.0) > sla_pct else "#4fc3f7",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                "yes" if r.get("healthy", False) else "no",
                                style={
                                    **td_base,
                                    "color": "#81c784" if r.get("healthy", False) else "#ef9a9a",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                f"{r.get('tp_suc_per_sec_robot', 0.0):.3f}",
                                style={**td_base, "color": "#4fc3f7", "fontWeight": 600},
                            ),
                            html.Td(
                                f"{r['avg_network_delay_ms']:.2f}",
                                style={**td_base, "color": "#4fc3f7", "fontWeight": 600},
                            ),
                        ]
                    )
                    for rid, r in sorted(robots.items())
                ]
            ),
        ],
    )


def _combined_task_episode_heatmap_fig(task_events: list[dict], task_progress: list[dict], title: str) -> go.Figure:
    fig = go.Figure()
    if not task_events and not task_progress:
        fig.update_layout(
            **_layout(
                title={"text": title, "font": {"size": 12, "color": "#888"}},
                xaxis={"title": "Episode"},
                yaxis={"title": "Task / Robot"},
                showlegend=False,
            )
        )
        return fig

    # Row key = "{robot_id}-{task_id} . averages"
    row_events: dict[str, dict[int, dict]] = {}
    row_total_episodes: dict[str, int] = {}

    for event in task_events:
        label = f"{event['robot_id']}-{int(event['task_id'])} . avg"
        row_events.setdefault(label, {})[int(event["episode_idx"])] = event
        row_total_episodes[label] = max(
            row_total_episodes.get(label, 0),
            int(event.get("total_episodes") or 0),
            int(event["episode_idx"]) + 1,
        )

    for prog in task_progress:
        label = f"{prog['robot_id']}-{int(prog['task_id'])} . avg"
        row_events.setdefault(label, {})
        row_total_episodes[label] = max(
            row_total_episodes.get(label, 0),
            int(prog.get("total_episodes") or 0),
            int(prog["episode_idx"]) + 1,
        )

    row_order = sorted(row_events.keys())
    max_episode = max(row_total_episodes.values(), default=1)
    row_count = len(row_order)
    row_ticktext: dict[str, str] = {}
    for label in row_order:
        completed = list(row_events[label].values())
        if completed:
            avg_duration_s = float(np.mean([float(event["duration_s"]) for event in completed]))
            avg_steps = float(np.mean([float(event["steps_taken"]) for event in completed]))
            row_ticktext[label] = (
                f"{label}  "
                f"<span style='color:#4fc3f7;font-weight:700'>{avg_duration_s:.1f}s</span> / "
                f"<span style='color:#ffb74d;font-weight:700'>{avg_steps:.0f}st</span>"
            )
        else:
            row_ticktext[label] = (
                f"{label}  <span style='color:#6b7280'>--s</span> / <span style='color:#6b7280'>--st</span>"
            )

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    hover_text: list[list[str]] = []
    border_x: list[int] = []
    border_y: list[str] = []
    border_color: list[str] = []

    for label in row_order:
        row_z: list[float | None] = []
        row_text: list[str] = []
        row_hover: list[str] = []
        for episode_idx in range(max_episode):
            if episode_idx >= row_total_episodes[label]:
                row_z.append(None)
                row_text.append("")
                row_hover.append("")
                continue

            event = row_events[label].get(episode_idx)
            border_x.append(episode_idx)
            border_y.append(label)
            if event is None:
                row_z.append(0.0)
                row_text.append("")
                row_hover.append(f"{label}<br>episode: {episode_idx}<br>status: pending")
                border_color.append("#374151")
                continue

            duration = float(event["duration_s"])
            steps = float(event["steps_taken"])
            row_z.append(2.0 if event["success"] else 1.0)
            row_text.append(f"{duration:.1f} s / {steps:.0f} st")
            row_hover.append(
                f"{label}<br>episode: {episode_idx}<br>duration: {duration:.3f} s<br>"
                f"steps: {steps:.0f} st<br>success: {event['success']}"
            )
            border_color.append("#22c55e" if event["success"] else "#ef4444")

        z.append(row_z)
        text.append(row_text)
        hover_text.append(row_hover)

    fig.add_trace(
        go.Heatmap(
            x=list(range(max_episode)),
            y=row_order,
            z=z,
            text=text,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#111"},
            zmin=0.0,
            zmax=2.0,
            colorscale=[
                [0.0, "#1f2937"],
                [0.333, "#1f2937"],
                [0.334, "#dc2626"],
                [0.666, "#dc2626"],
                [0.667, "#22c55e"],
                [1.0, "#22c55e"],
            ],
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )

    cell_size = max(14, min(46, int(760 / max(max_episode, row_count, 1))))
    fig.add_trace(
        go.Scatter(
            x=border_x,
            y=border_y,
            mode="markers",
            hoverinfo="skip",
            marker={
                "symbol": "square-open",
                "size": cell_size,
                "color": "rgba(0,0,0,0)",
                "line": {"width": 2, "color": border_color},
            },
            showlegend=False,
        )
    )

    fig.update_layout(
        **_layout(
            title={"text": title, "font": {"size": 12, "color": "#888"}},
            xaxis={"title": "Episode", "dtick": 1},
            yaxis={
                "title": "Robot / Task",
                "categoryorder": "array",
                "categoryarray": row_order,
                "tickmode": "array",
                "tickvals": row_order,
                "ticktext": [row_ticktext[row] for row in row_order],
                "autorange": "reversed",
                "ticklabelstandoff": 20,
                "automargin": True,
            },
            showlegend=False,
            height=max(440, row_count * 56 + 120),
            margin={"t": 40, "r": 18, "b": 70, "l": 430},
        )
    )
    return fig


def _sla_capacity_curve_fig(sla_capacity_curve: list[dict], sla_pct: float) -> go.Figure:
    fig = go.Figure()
    if sla_capacity_curve:
        xs = [point["sla_pct"] for point in sla_capacity_curve]
        ys = [point["healthy_robot_count"] for point in sla_capacity_curve]
        active = [point["active_robot_count"] for point in sla_capacity_curve]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"color": "#ce93d8", "width": 2},
                marker={"size": 5},
                customdata=active,
                hovertemplate="SLA %{x:.0f}%<br>healthy %{y} / %{customdata}<extra></extra>",
                name="healthy robots",
            )
        )
        fig.add_vline(x=sla_pct, line_width=1, line_dash="dot", line_color="#4fc3f7")
    fig.update_layout(
        **_layout(
            title={"text": "SLA Capacity Curve", "font": {"size": 12, "color": "#888"}},
            xaxis={"title": "SLA threshold (%)", "range": [0, 100], "dtick": 10},
            yaxis={"title": "Healthy robots"},
            height=320,
            showlegend=False,
        )
    )
    return fig


def _healthy_robots_over_time_fig(series: list[dict], sla_pct: float) -> go.Figure:
    fig = go.Figure()
    if series:
        xs = [point["t"] for point in series]
        healthy = [point["healthy_robot_count"] for point in series]
        active = [point["active_robot_count"] for point in series]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=healthy,
                mode="lines+markers",
                line={"color": "#81c784", "width": 2},
                marker={"size": 4},
                name="healthy robots",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=active,
                mode="lines",
                line={"color": "#607d8b", "width": 1.5, "dash": "dot"},
                name="active robots",
            )
        )
    fig.update_layout(
        **_layout(
            title={"text": f"Healthy Robots Over Time @ {sla_pct:.0f}% SLA", "font": {"size": 12, "color": "#888"}},
            xaxis={"title": "Time since server start (s)"},
            yaxis={"title": "Robots"},
            height=320,
            showlegend=True,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4fc3f7",
    "#81c784",
    "#ff8a65",
    "#ce93d8",
    "#ffb74d",
    "#f06292",
    "#4db6ac",
    "#aed581",
    "#7986cb",
    "#4dd0e1",
]


def _gpu_dist_fig(batches: list[dict]) -> go.Figure:
    fig = go.Figure()
    by_size: dict[int, list[float]] = {}
    for b in batches:
        by_size.setdefault(b["batch_size"], []).append(b["gpu_time_ms"])
    for i, s in enumerate(sorted(by_size)):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Histogram(
                x=by_size[s],
                name=str(s),
                nbinsx=60,
                marker_color=color,
                marker_opacity=0.6,
                legendgroup=str(s),
            )
        )
    fig.update_layout(
        **_layout(
            barmode="overlay",
            xaxis={"title": "GPU time (ms)"},
            yaxis={"title": "Count"},
            height=320,
            showlegend=True,
            legend={"title": {"text": "Batch size"}, **_DARK["legend"]},
        )
    )
    return fig


def _batch_fig(batches: list[dict]) -> go.Figure:
    fig = go.Figure()
    if batches:
        times = [b["t"] for b in batches]
        sizes = [b["batch_size"] for b in batches]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=sizes,
                mode="markers",
                name="batch size",
                marker={"size": 4, "color": "#ce93d8", "opacity": 0.6},
            )
        )
    fig.update_layout(**_layout(xaxis={"title": "Time since server start (s)"}, yaxis={"title": "Batch size"}))
    return fig


def _busy_fig(batches: list[dict]) -> go.Figure:
    fig = go.Figure()
    if len(batches) >= 2:
        t0 = batches[0]["inference_start_t"]
        t1 = batches[-1]["inference_end_t"]
        n = int(t1 - t0) + 1
        pct = [0.0] * n
        for b in batches:
            s = b["inference_start_t"] - t0
            e = b["inference_end_t"] - t0
            lo, hi = int(s), min(int(e), n - 1)
            for k in range(lo, hi + 1):
                ov = min(e, k + 1) - max(s, k)
                if ov > 0:
                    pct[k] += ov * 100
        times = np.array([t0 + i + 0.5 for i in range(n)], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=pct,
                mode="lines",
                fill="tozeroy",
                line={"color": "#4fc3f7", "width": 1.5},
                fillcolor="rgba(79,195,247,0.12)",
                name="busy %",
            )
        )
    fig.update_layout(
        **_layout(xaxis={"title": "Time since server start (s)"}, yaxis={"title": "GPU busy (%)", "range": [0, 100]})
    )
    return fig


def _gantt_fig(
    batches: list[dict],
    window_s: float,
    replan_markers: list[dict],
    kickoff_markers: list[dict],
) -> go.Figure:
    fig = go.Figure()
    if not batches:
        return fig
    max_t = batches[-1]["t"]
    visible = [b for b in batches if b["t"] >= max_t - window_s]
    visible_min_t = max_t - window_s
    all_robots = sorted({rid for b in visible for rid in b["robot_ids"]})
    palette = [
        "#4fc3f7",
        "#81c784",
        "#ff8a65",
        "#ce93d8",
        "#ffb74d",
        "#f06292",
        "#4db6ac",
        "#aed581",
        "#7986cb",
        "#4dd0e1",
    ]
    rc = {r: palette[i % len(palette)] for i, r in enumerate(all_robots)}
    plan_palette = [
        "#ff8a65",
        "#81c784",
        "#4fc3f7",
        "#ce93d8",
        "#ffb74d",
        "#f06292",
        "#4db6ac",
        "#aed581",
        "#7986cb",
        "#4dd0e1",
    ]

    def _plan_color(plan_index: int | None) -> str:
        if plan_index is None:
            return "#546e7a"
        return plan_palette[plan_index % len(plan_palette)]

    tmap: dict[str, dict] = {}
    for b in visible:
        dur = b["inference_end_t"] - b["inference_start_t"]
        border_color = _plan_color(b.get("plan_index"))
        for rid in b["robot_ids"]:
            if rid not in tmap:
                tmap[rid] = {"x": [], "base": [], "y": [], "color": rc[rid], "line_color": []}
            tmap[rid]["x"].append(dur)
            tmap[rid]["base"].append(b["inference_start_t"])
            tmap[rid]["y"].append(rid)
            tmap[rid]["line_color"].append(border_color)
    for rid, td in sorted(tmap.items()):
        fig.add_trace(
            go.Bar(
                x=td["x"],
                base=td["base"],
                y=td["y"],
                orientation="h",
                name=rid,
                marker={
                    "color": td["color"],
                    "line": {"width": 2, "color": td["line_color"]},
                },
            )
        )
    for marker in replan_markers:
        t = marker["t"]
        if visible_min_t <= t <= max_t:
            fig.add_vline(
                x=t,
                line_width=1.5,
                line_dash="solid",
                line_color=_plan_color(marker.get("plan_index")),
            )
    for marker in kickoff_markers:
        t = marker["t"]
        if visible_min_t <= t <= max_t:
            # Draw kickoff as a short tick near the x-axis to reduce timeline clutter.
            fig.add_shape(
                type="line",
                x0=t,
                x1=t,
                y0=0.0,
                y1=0.04,
                xref="x",
                yref="paper",
                line={"width": 2.0, "color": _plan_color(marker.get("plan_index"))},
            )
    fig.update_layout(
        **_layout(
            barmode="overlay",
            height=max(200, len(all_robots) * 32 + 80),
            xaxis={"title": "Time since server start (s)"},
            yaxis={
                "categoryorder": "array",
                "categoryarray": all_robots,
                "autorange": "reversed",
            },
            showlegend=False,
            margin={**_DARK["margin"], "l": 100},
        )
    )
    return fig


def _actions_left_heatmap_fig(robot_actions_left: dict[str, tuple[np.ndarray, np.ndarray]]) -> go.Figure:
    """Scatter of actions_left over time. Robots on y-axis, time on x-axis (aligned with batch/busy/gantt)."""
    robots = sorted(robot_actions_left.keys())
    fig = go.Figure()
    if not robots:
        fig.update_layout(**_layout(xaxis={"title": "Time since server start (s)"}, yaxis={"title": "Robot"}))
        return fig

    row_h = 20
    height = max(180, len(robots) * row_h + 80)
    marker_size = row_h - 2

    for i, rid in enumerate(robots):
        times, values = robot_actions_left[rid]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[rid] * len(times),
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": marker_size,
                    "color": values,
                    "colorscale": "RdYlGn",
                    "cmin": 0,
                    "showscale": i == len(robots) - 1,
                    "colorbar": {"title": "Actions left", "thickness": 12},
                },
                name=rid,
                showlegend=False,
            )
        )
    fig.update_layout(
        **_layout(
            xaxis={"title": "Time since server start (s)"},
            yaxis={
                "title": "Robot",
                "categoryorder": "array",
                "categoryarray": robots,
                "autorange": "reversed",
            },
            height=height,
            margin={**_DARK["margin"], "l": 100},
        )
    )
    return fig


def _stage_figs(snap: Snapshot, robot: str) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    inbound, queue_, infer_, outbound = [], [], [], []
    for b in snap.batch_history:
        for req in b["per_request"]:
            if robot != "all" and req["robot_id"] != robot:
                continue
            inbound.append(req["inbound_ms"])
            queue_.append(req["queue_ms"])
            infer_.append(req["infer_ms"])
    od = snap.outbound_delays_ms
    outbound = list(od.get(robot) or []) if robot != "all" else [d for v in od.values() for d in v]

    small = {**_DARK, "margin": {"t": 36, "r": 8, "b": 44, "l": 48}, "height": 280}
    figs = []
    for data, color, title in [
        (inbound, "#4fc3f7", "Inbound Network"),
        (queue_, "#ff8a65", "Queue Wait"),
        (infer_, "#81c784", "Inference"),
        (outbound, "#ce93d8", "Outbound Network"),
    ]:
        f = go.Figure()
        if data:
            f.add_trace(go.Histogram(x=data, nbinsx=100, marker_color=color, marker_opacity=0.85, name=title))
        f.update_layout(
            **{
                **small,
                "title": {"text": title, "font": {"size": 12, "color": "#888"}},
                "xaxis": {**_DARK["xaxis"], "title": "ms"},
                "yaxis": {**_DARK["yaxis"], "title": "count"},
            }
        )
        figs.append(f)
    return tuple(figs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_dash_app(metadata: ServerMetadata, metrics_store: MetricsStore) -> dash.Dash:
    app = dash.Dash(
        __name__,
        url_base_pathname="/",
        title="openpi · metrics",
        prevent_initial_callbacks="initial_duplicate",
        suppress_callback_exceptions=True,
    )

    meta_line = (
        f"config: {metadata.config_name}  ·  env: {metadata.env}  ·  "
        f"max_batch: {metadata.max_batch_size}  ·  "
        f"location: {metadata.location or 'unknown'}  ·  "
        f"checkpoint: {metadata.checkpoint_dir}"
    )

    app.layout = html.Div(
        style=_BODY,
        children=[
            dcc.Store(id="store-xrange"),
            dcc.Store(id="store-auto-refresh-enabled", data=False),
            dcc.Interval(id="interval-refresh", interval=5000, disabled=True, n_intervals=0),
            # Header
            html.H1(
                "openpi · metrics",
                style={"fontSize": "1.3em", "fontWeight": 700, "color": "#fff", "marginBottom": "2px"},
            ),
            html.Div(id="div-subtitle", style={"color": "#555", "fontSize": "0.8em", "marginBottom": "4px"}),
            html.Div(
                meta_line,
                style={"color": "#444", "fontSize": "0.75em", "marginBottom": "20px", "fontFamily": "inherit"},
            ),
            # Toolbar
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "24px",
                },
                children=[
                    html.Button("⟳ Refresh", id="btn-refresh", n_clicks=0, style=_BTN_PRIMARY),
                    html.Button("▶ Auto", id="btn-auto-refresh", n_clicks=0, style=_BTN_PRIMARY),
                    html.Label("every", style={"color": "#555", "fontSize": "0.85em"}),
                    dcc.Dropdown(
                        id="dd-auto-refresh-seconds",
                        options=[
                            {"label": "1", "value": 1},
                            {"label": "2", "value": 2},
                            {"label": "5", "value": 5},
                            {"label": "10", "value": 10},
                            {"label": "15", "value": 15},
                            {"label": "30", "value": 30},
                            {"label": "60", "value": 60},
                        ],
                        value=5,
                        clearable=False,
                        searchable=False,
                        style={"width": "84px", "fontSize": "0.9em"},
                    ),
                    html.Label("seconds", style={"color": "#555", "fontSize": "0.85em"}),
                    html.Label("Last", style={"color": "#555", "fontSize": "0.85em"}),
                    dcc.Input(
                        id="input-window",
                        type="number",
                        placeholder="all",
                        min=1,
                        debounce=True,
                        style={**_INPUT, "width": "80px", "padding": "5px 8px"},
                    ),
                    html.Label("seconds", style={"color": "#555", "fontSize": "0.85em"}),
                    html.Span(id="span-status", style={"color": "#555", "fontSize": "0.8em", "marginLeft": "4px"}),
                ],
            ),
            # Summary
            _section(
                "Summary",
                html.Div(
                    id="div-stats",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(150px, 1fr))",
                        "gap": "8px",
                    },
                ),
            ),
            # Per-Robot
            _section("Per-Robot", html.Div(id="div-robots")),
            # Charts
            _section(
                "Charts",
                # Combined episode heatmap
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("Episode Progress Heatmap", style=_CARD_HDR),
                        dcc.Graph(id="graph-task-heatmap", config=_CFG),
                    ],
                ),
                # GPU inference dist
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Inference Time by Batch Size", style=_CARD_HDR),
                        dcc.Graph(id="graph-gpu-dist", config=_CFG, style={"height": "320px"}),
                    ],
                ),
                # Stage latency
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div(
                            style=_CARD_HDR,
                            children=[
                                html.Span("Stage Latency Distributions"),
                                html.Label("Robot:", style={"color": "#555"}),
                                dcc.Dropdown(
                                    id="dd-robot",
                                    options=[{"label": "all", "value": "all"}],
                                    value="all",
                                    clearable=False,
                                    style={"width": "160px", "fontSize": "0.9em"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)"},
                            children=[
                                dcc.Graph(id="graph-inbound", config=_CFG, style={"height": "280px"}),
                                dcc.Graph(id="graph-queue", config=_CFG, style={"height": "280px"}),
                                dcc.Graph(id="graph-infer", config=_CFG, style={"height": "280px"}),
                                dcc.Graph(id="graph-outbound", config=_CFG, style={"height": "280px"}),
                            ],
                        ),
                    ],
                ),
                # SLA capacity curve
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("SLA Capacity Curve", style=_CARD_HDR),
                        html.Div(
                            style={"padding": "10px 14px 0 14px"},
                            children=[
                                html.Div(
                                    "SLA threshold (%)",
                                    style={"color": "#666", "fontSize": "0.8em", "marginBottom": "4px"},
                                ),
                                dcc.Slider(
                                    id="slider-sla-pct",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=10,
                                    marks={
                                        0: {"label": "0%", "style": {"color": "#fff"}},
                                        25: {"label": "25%", "style": {"color": "#fff"}},
                                        50: {"label": "50%", "style": {"color": "#fff"}},
                                        75: {"label": "75%", "style": {"color": "#fff"}},
                                        100: {"label": "100%", "style": {"color": "#fff"}},
                                    },
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="graph-sla-capacity", config=_CFG, style={"height": "320px"}),
                    ],
                ),
                # Healthy robots over time
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("Healthy Robots Over Time", style=_CARD_HDR),
                        dcc.Graph(id="graph-healthy-robots", config=_CFG, style={"height": "320px"}),
                    ],
                ),
                # Batch sizes (resampled)
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("Batch Sizes Over Time", style=_CARD_HDR),
                        dcc.Graph(id="graph-batch", config=_CFG, style={"height": "320px"}),
                    ],
                ),
                # GPU busy
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Busy (%)", style=_CARD_HDR),
                        dcc.Graph(id="graph-busy", config=_CFG, style={"height": "320px"}),
                    ],
                ),
                # Gantt
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Gantt", style=_CARD_HDR),
                        dcc.Graph(id="graph-gantt", config=_CFG),
                    ],
                ),
                # Actions left heatmap
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("Actions Left (server-side)", style=_CARD_HDR),
                        dcc.Graph(id="graph-actions-left", config=_CFG),
                    ],
                ),
            ),
        ],
    )

    @app.callback(
        Output("store-auto-refresh-enabled", "data"),
        Output("btn-auto-refresh", "children"),
        Output("btn-auto-refresh", "style"),
        Output("interval-refresh", "disabled"),
        Output("interval-refresh", "interval"),
        Input("btn-auto-refresh", "n_clicks"),
        Input("dd-auto-refresh-seconds", "value"),
        State("store-auto-refresh-enabled", "data"),
    )
    def _configure_auto_refresh(
        n_clicks: int,
        seconds: int | None,
        enabled: bool | None,
    ) -> tuple[bool, str, dict, bool, int]:
        _ = n_clicks
        is_enabled = bool(enabled)
        if ctx.triggered_id == "btn-auto-refresh":
            is_enabled = not is_enabled
        refresh_seconds = max(1, int(seconds) if seconds is not None else 5)
        btn_label = f"⏸ Auto ({refresh_seconds}s)" if is_enabled else "▶ Auto"
        btn_style = _BTN_ACTIVE if is_enabled else _BTN_PRIMARY
        return is_enabled, btn_label, btn_style, (not is_enabled), refresh_seconds * 1000

    # -------------------------------------------------------------------------
    # Main refresh: stats, robots, GPU dist, Gantt
    # -------------------------------------------------------------------------
    @app.callback(
        Output("div-subtitle", "children"),
        Output("div-stats", "children"),
        Output("div-robots", "children"),
        Output("graph-gpu-dist", "figure"),
        Output("graph-task-heatmap", "figure"),
        Output("graph-sla-capacity", "figure"),
        Output("graph-healthy-robots", "figure"),
        Output("graph-gantt", "figure"),
        Output("dd-robot", "options"),
        Output("span-status", "children"),
        Input("btn-refresh", "n_clicks"),
        Input("interval-refresh", "n_intervals"),
        Input("slider-sla-pct", "value"),
        State("input-window", "value"),
    )
    def _refresh_main(
        n_clicks: int,
        n_intervals: int,
        sla_pct: float | None,
        window_s: float | None,
    ) -> tuple:
        _ = n_clicks, n_intervals
        sla_pct = float(sla_pct) if sla_pct is not None else 10.0
        snap = metrics_store.snapshot(window_s, sla_pct=sla_pct)
        batches = snap.batch_history

        def f(v: float) -> str:
            return f"{v:.1f}"

        subtitle = f"uptime {f(snap.uptime_s)}s · {snap.total_requests:,} requests · {snap.total_batches:,} batches"

        stats = [
            ("total requests", f"{snap.total_requests:,}"),
            ("total responses", f"{snap.total_responses:,}"),
            ("requests / s", f(snap.requests_per_second)),
            ("responses / s", f(snap.responses_per_second)),
            ("p50 latency (ms)", f(snap.p50_latency_ms)),
            ("p99 latency (ms)", f(snap.p99_latency_ms)),
            ("avg GPU time (ms)", f(snap.avg_gpu_time_ms)),
            ("avg queue delay (ms)", f(snap.avg_queue_delay_ms)),
            ("total batches", f"{snap.total_batches:,}"),
            ("task success (%)", f(snap.task_success_rate_pct)),
            ("TP (suc/sec/all)", f"{snap.tp_suc_per_sec_all:.3f}"),
        ]
        stat_cards = [_stat_card(v, lbl) for lbl, v in stats]

        robots = snap.per_robot
        robot_el = _robot_table(robots, sla_pct)

        gpu_fig = _gpu_dist_fig(batches)
        task_heatmap_fig = _combined_task_episode_heatmap_fig(
            snap.task_events,
            snap.task_progress,
            title="Success, Completion Time Metrics",
        )
        sla_capacity_fig = _sla_capacity_curve_fig(snap.sla_capacity_curve, sla_pct)
        healthy_robots_fig = _healthy_robots_over_time_fig(snap.healthy_robots_over_time, sla_pct)
        gantt = _gantt_fig(
            batches,
            float(window_s) if window_s else float("inf"),
            snap.replan_markers,
            snap.kickoff_markers,
        )

        robot_opts = [{"label": "all", "value": "all"}] + [{"label": rid, "value": rid} for rid in sorted(robots)]
        status = "last refresh: " + datetime.datetime.now(datetime.UTC).astimezone().strftime("%H:%M:%S")

        return (
            subtitle,
            stat_cards,
            robot_el,
            gpu_fig,
            task_heatmap_fig,
            sla_capacity_fig,
            healthy_robots_fig,
            gantt,
            robot_opts,
            status,
        )

    # -------------------------------------------------------------------------
    # Batch sizes
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-batch", "figure"),
        Input("btn-refresh", "n_clicks"),
        Input("interval-refresh", "n_intervals"),
        State("input-window", "value"),
    )
    def _load_batch(
        n_clicks: int,
        n_intervals: int,
        window_s: float | None,
    ) -> go.Figure:
        _ = n_clicks, n_intervals
        batches = metrics_store.snapshot(window_s).batch_history

        fig = go.Figure()
        if batches:
            times = np.array([b["t"] for b in batches], dtype=float)
            sizes = np.array([b["batch_size"] for b in batches], dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=sizes,
                    mode="markers",
                    name="batch size",
                    marker={"size": 4, "color": "#ce93d8", "opacity": 0.6},
                )
            )
        fig.update_layout(
            **_layout(
                xaxis={"title": "Time since server start (s)"},
                yaxis={"title": "Batch size"},
                height=320,
                margin={**_DARK["margin"], "l": 100},
            )
        )
        return fig

    # -------------------------------------------------------------------------
    # GPU busy
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-busy", "figure"),
        Input("btn-refresh", "n_clicks"),
        Input("interval-refresh", "n_intervals"),
        State("input-window", "value"),
    )
    def _load_busy(
        n_clicks: int,
        n_intervals: int,
        window_s: float | None,
    ) -> go.Figure:
        _ = n_clicks, n_intervals
        batches = metrics_store.snapshot(window_s).batch_history

        fig = go.Figure()
        if len(batches) >= 2:
            t0 = batches[0]["inference_start_t"]
            t1 = batches[-1]["inference_end_t"]
            n = int(t1 - t0) + 1
            pct = [0.0] * n
            for b in batches:
                s = b["inference_start_t"] - t0
                e = b["inference_end_t"] - t0
                lo, hi = int(s), min(int(e), n - 1)
                for k in range(lo, hi + 1):
                    ov = min(e, k + 1) - max(s, k)
                    if ov > 0:
                        pct[k] += ov * 100
            times = np.array([t0 + i + 0.5 for i in range(n)], dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.array(pct, dtype=float),
                    mode="lines",
                    fill="tozeroy",
                    line={"color": "#4fc3f7", "width": 1.5},
                    fillcolor="rgba(79,195,247,0.12)",
                    name="busy %",
                )
            )
        fig.update_layout(
            **_layout(
                xaxis={"title": "Time since server start (s)"},
                yaxis={"title": "GPU busy (%)", "range": [0, 100]},
                height=320,
                margin={**_DARK["margin"], "l": 100},
            )
        )
        return fig

    # -------------------------------------------------------------------------
    # X-axis sync: batch / busy / gantt / actions-left share the same time axis
    # -------------------------------------------------------------------------
    @app.callback(
        Output("store-xrange", "data"),
        Input("graph-batch", "relayoutData"),
        Input("graph-busy", "relayoutData"),
        Input("graph-gantt", "relayoutData"),
        Input("graph-actions-left", "relayoutData"),
        prevent_initial_call=True,
    )
    def _capture_xrange(
        batch_relay: dict | None,
        busy_relay: dict | None,
        gantt_relay: dict | None,
        actions_relay: dict | None,
    ) -> list | None:
        relay_map = {
            "graph-batch": batch_relay,
            "graph-busy": busy_relay,
            "graph-gantt": gantt_relay,
            "graph-actions-left": actions_relay,
        }
        relay = relay_map.get(ctx.triggered_id)
        if relay and "xaxis.range[0]" in relay:
            return [relay["xaxis.range[0]"], relay["xaxis.range[1]"]]
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("graph-batch", "figure", allow_duplicate=True),
        Output("graph-busy", "figure", allow_duplicate=True),
        Output("graph-gantt", "figure", allow_duplicate=True),
        Output("graph-actions-left", "figure", allow_duplicate=True),
        Input("store-xrange", "data"),
        prevent_initial_call=True,
    )
    def _apply_xrange(xrange: list | None) -> tuple:
        if not xrange:
            raise dash.exceptions.PreventUpdate
        p_batch = Patch()
        p_batch["layout"]["xaxis"]["range"] = xrange
        p_busy = Patch()
        p_busy["layout"]["xaxis"]["range"] = xrange
        p_gantt = Patch()
        p_gantt["layout"]["xaxis"]["range"] = xrange
        p_actions = Patch()
        p_actions["layout"]["xaxis"]["range"] = xrange
        return p_batch, p_busy, p_gantt, p_actions

    # -------------------------------------------------------------------------
    # Stage latency distributions
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-inbound", "figure"),
        Output("graph-queue", "figure"),
        Output("graph-infer", "figure"),
        Output("graph-outbound", "figure"),
        Input("btn-refresh", "n_clicks"),
        Input("interval-refresh", "n_intervals"),
        Input("dd-robot", "value"),
        State("input-window", "value"),
    )
    def _update_stage(
        n_clicks: int,
        n_intervals: int,
        robot: str,
        window_s: float | None,
    ) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        _ = n_clicks, n_intervals
        return _stage_figs(metrics_store.snapshot(window_s), robot or "all")

    # Histogram zoom: refine bin size when user zooms in
    def _register_histogram_zoom(gid: str) -> None:
        @app.callback(
            Output(gid, "figure", allow_duplicate=True),
            Input(gid, "relayoutData"),
            State(gid, "figure"),
            prevent_initial_call=True,
        )
        def _zoom_histogram(relay: dict | None, figure: dict | None) -> Patch:
            if not relay or "xaxis.range[0]" not in relay:
                raise dash.exceptions.PreventUpdate
            if not figure or not figure.get("data"):
                raise dash.exceptions.PreventUpdate
            x0 = float(relay["xaxis.range[0]"])
            x1 = float(relay["xaxis.range[1]"])
            if x1 <= x0:
                raise dash.exceptions.PreventUpdate
            p = Patch()
            p["data"][0]["xbins"]["size"] = (x1 - x0) / 150
            return p

    for _gid in ["graph-inbound", "graph-queue", "graph-infer", "graph-outbound"]:
        _register_histogram_zoom(_gid)

    # -------------------------------------------------------------------------
    # Actions left heatmap
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-actions-left", "figure"),
        Input("btn-refresh", "n_clicks"),
        Input("interval-refresh", "n_intervals"),
        State("input-window", "value"),
    )
    def _load_actions_left(
        n_clicks: int,
        n_intervals: int,
        window_s: float | None,
    ) -> go.Figure:
        _ = n_clicks, n_intervals
        snap = metrics_store.snapshot(window_s)
        return _actions_left_heatmap_fig(snap.robot_actions_left)

    return app
