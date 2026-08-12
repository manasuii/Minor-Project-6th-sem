import matplotlib
matplotlib.use("TkAgg")

import os
import time
import warnings
from collections import deque

import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.transforms import Affine2D

# ── project imports (real classes, exactly as the codebase defines them) ──
from quadrotor_simulator import simulate_step, PARAMS
from mpc_surrogate import SurrogateMPC, N_HORIZON, U_MAX, U_MIN
from baseline_controllers import PIDController, LinearisedMPC, ODEMPCController

warnings.filterwarnings("ignore")

DT = PARAMS["dt"]                 # 0.05 s
WINDOW_S = 8.0                    # rolling display / metric window
W = int(round(WINDOW_S / DT))     # 160 samples
SOLVE_AVG_N = 10                  # rolling solve-time average length

# Controller order + colours — the defense palette.
COLORS = {
    "Surrogate-MPC": "#185FA5",   # blue
    "ODE-MPC":       "#534AB7",   # purple
    "Lin-MPC":       "#0F6E56",   # green
    "PID":           "#854F0B",   # amber
    "ref":           "#A32D2D",   # red dashed
}
ORDER = ["Surrogate-MPC", "ODE-MPC", "Lin-MPC", "PID"]

CLIP_LO = np.asarray(U_MIN, dtype=float)
CLIP_HI = np.asarray(U_MAX, dtype=float)

# ── attitude front-view geometry (ported from drawQuad, y flipped to y-up) ──
ATT_W, ATT_H = 210.0, 96.0
ATT_CX, ATT_CY = 105.0, 54.0      # SVG cy=42 from top -> 96-42 = 54 from bottom
ATT_ARM, ATT_PR = 60.0, 14.0      # arm half-length, propeller rx
ATT_GROUND_Y = 12.0               # SVG y=84 from top -> 96-84 = 12 from bottom
ATT_ARM_COL  = "#334155"
ATT_BODY_COL = "#1E293B"
ATT_HUB_COL  = "#475569"
ATT_GND_COL  = "#CBD5E1"
# Tilt sense: SVG rotates clockwise on screen for +φ. In y-up matplotlib that is
# rotate_deg_around(cx, cy, -φ_deg). Flip TILT_SIGN if it should mirror.
TILT_SIGN = -1.0


# ─────────────────────────────────────────
#  Reference signal (as specified)
# ─────────────────────────────────────────
def ref_signal(t, scenario, ref_deg):
    r = ref_deg * np.pi / 180.0
    if scenario == "step":
        return np.array([r, 0, 0, 0, 0, 0])
    if scenario == "sine":
        return np.array([r * np.sin(0.5 * np.pi * t), 0, 0, 0, 0, 0])
    if scenario == "hover":
        return np.zeros(6)
    return np.zeros(6)


# ─────────────────────────────────────────
#  Rolling-window metrics (per the defense spec)
# ─────────────────────────────────────────
def compute_metrics(phi_hist, ref_hist):
    """phi_hist / ref_hist: 1-D arrays of equal length over the rolling window."""
    if len(phi_hist) == 0:
        return 0.0, 0.0, 0.0
    phi = np.asarray(phi_hist)
    ref = np.asarray(ref_hist)
    phi_ref = float(ref[-1])                       # current target (scalar)

    # RMSE — element-wise vs the reference history (reduces exactly to the
    # spec's scalar formula for step/hover, stays correct for sine).
    rmse = float(np.sqrt(np.mean((phi - ref) ** 2)))

    # Settling time (2% criterion, small absolute floor)
    tol = max(0.02 * abs(phi_ref), 0.003)
    window_duration = len(phi) * DT
    settle = 0.0
    for i, v in enumerate(reversed(phi)):
        if abs(v - phi_ref) > tol:
            settle = window_duration - i * DT
            break

    # Overshoot (only for a non-trivial positive target)
    if abs(phi_ref) > 0.01:
        overshoot = max(0.0, (phi.max() - phi_ref) / abs(phi_ref) * 100.0)
    else:
        overshoot = 0.0
    return rmse, settle, overshoot


# ═════════════════════════════════════════
#  Simulator
# ═════════════════════════════════════════
class LiveSimulator:
    def __init__(self):
        # ── build the four controllers EXACTLY as run_experiments.py does ──
        print("Initialising controllers (loads the ResNN into Surrogate-MPC and "
              "compiles the CasADi solvers)...")
        self.controllers = {
            "Surrogate-MPC": SurrogateMPC(N=N_HORIZON),
            "ODE-MPC":       ODEMPCController(N=N_HORIZON),
            "Lin-MPC":       LinearisedMPC(N=N_HORIZON),
            "PID":           PIDController(),
        }
        self._warmup()

        # ── per-controller live state ──
        self.x = {k: np.zeros(6) for k in ORDER}
        self.u_prev = {k: np.zeros(3) for k in ORDER}
        self.phi_hist = {k: deque(maxlen=W) for k in ORDER}
        self.tau_hist = {k: deque(maxlen=W) for k in ORDER}   # tau_phi, mN·m
        self.solve_ms = {k: deque(maxlen=SOLVE_AVG_N) for k in ORDER}

        # ── shared clock + reference history (identical ref for all four) ──
        self.t_hist = deque(maxlen=W)
        self.ref_hist = deque(maxlen=W)
        self.sim_t = 0.0

        # ── control settings ──
        self.scenario = "step"
        self.ref_deg = 20.0
        self.speed = 1
        self.running = True

        self._build_gui()

    # ──────────────────────────────────────
    def _warmup(self):
        x0, ref = np.zeros(6), np.array([0.2, 0, 0, 0, 0, 0])
        for name, c in self.controllers.items():
            if hasattr(c, "solve"):
                for _ in range(3):                      # same warmup as the codebase
                    c.solve(x0, ref)

    def reset_states(self):
        self.sim_t = 0.0
        self.t_hist.clear()
        self.ref_hist.clear()
        for k in ORDER:
            self.x[k] = np.zeros(6)
            self.u_prev[k] = np.zeros(3)
            self.phi_hist[k].clear()
            self.tau_hist[k].clear()
            self.solve_ms[k].clear()
            c = self.controllers[k]
            if hasattr(c, "reset"):
                c.reset()

    # ──────────────────────────────────────
    def _step_controller(self, name, x, ref):
        """One controller solve + one plant step. Returns (x_next, tau_phi_mNm, ms)."""
        c = self.controllers[name]
        t0 = time.perf_counter()
        try:
            if hasattr(c, "solve"):
                u, _ms, _ok = c.solve(x, ref)
            else:                                       # PID -> compute(), no tuple
                u = c.compute(x, ref)
            ms = (time.perf_counter() - t0) * 1000.0
            u = np.asarray(u, dtype=float).reshape(3)
            if not np.all(np.isfinite(u)):
                raise ValueError("non-finite control")
        except Exception as exc:                        # CasADi or any solver raise
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"[warn] {name} solve failed ({exc}); reusing previous u")
            u = self.u_prev[name].copy()

        u = np.clip(u, CLIP_LO, CLIP_HI)                # same clip as run_experiments
        self.u_prev[name] = u
        x_next = simulate_step(x, u)                    # same plant step as the codebase
        return x_next, float(u[0]) * 1000.0, ms         # tau_phi in mN·m

    def sim_substep(self):
        ref = ref_signal(self.sim_t, self.scenario, self.ref_deg)
        self.t_hist.append(self.sim_t)
        self.ref_hist.append(float(ref[0]))
        for name in ORDER:
            x_next, tau_phi_mNm, ms = self._step_controller(name, self.x[name], ref)
            self.x[name] = x_next
            self.phi_hist[name].append(float(x_next[0]))
            self.tau_hist[name].append(tau_phi_mNm)
            self.solve_ms[name].append(ms)
        self.sim_t += DT

    # ══════════════════════════════════════
    #  Attitude front-view (port of HTML drawQuad)
    # ══════════════════════════════════════
    def _make_attitude_axis(self, ax, name, col):
        ax.set_xlim(0, ATT_W)
        ax.set_ylim(0, ATT_H)
        ax.set_aspect("equal")
        ax.axis("off")

        # static ground reference line
        ax.plot([14, ATT_W - 14], [ATT_GROUND_Y, ATT_GROUND_Y],
                color=ATT_GND_COL, lw=1.0, solid_capstyle="round", zorder=1)

        cx, cy, arm, pr = ATT_CX, ATT_CY, ATT_ARM, ATT_PR
        # rotating parts (created at φ=0, transform updated each frame)
        arm_line, = ax.plot([cx - arm, cx + arm], [cy, cy],
                            color=ATT_ARM_COL, lw=3.0, solid_capstyle="round", zorder=3)
        body  = Rectangle((cx - 8, cy - 6), 16, 12, facecolor=ATT_BODY_COL,
                          edgecolor="none", zorder=4)
        propL = Ellipse((cx - arm, cy), 2 * pr, 2 * 4.5, facecolor=col,
                        edgecolor="none", alpha=0.82, zorder=5)
        propR = Ellipse((cx + arm, cy), 2 * pr, 2 * 4.5, facecolor=col,
                        edgecolor="none", alpha=0.82, zorder=5)
        hubL  = Circle((cx - arm, cy), 3, facecolor=ATT_HUB_COL, edgecolor="none", zorder=6)
        hubR  = Circle((cx + arm, cy), 3, facecolor=ATT_HUB_COL, edgecolor="none", zorder=6)
        ctr   = Circle((cx, cy), 3.6, facecolor="#ffffff", edgecolor=ATT_BODY_COL,
                       lw=1.4, zorder=7)
        for p in (body, propL, propR, hubL, hubR, ctr):
            ax.add_patch(p)

        # labels: name (left) + live angle (right)
        ax.text(0.0, 1.04, name, color=col, fontsize=9.5, fontweight="bold",
                va="bottom", ha="left", transform=ax.transAxes)
        angle_txt = ax.text(1.0, 1.04, "0.0°", color="#334155", fontsize=10.5,
                            family="monospace", va="bottom", ha="right",
                            transform=ax.transAxes)

        self.att_parts[name] = [arm_line, body, propL, propR, hubL, hubR, ctr]
        self.att_angle[name] = angle_txt
        self.att_ax[name] = ax

    def _update_attitude(self, name, phi_rad):
        deg = phi_rad * 180.0 / np.pi
        ax = self.att_ax[name]
        rot = (Affine2D().rotate_deg_around(ATT_CX, ATT_CY, TILT_SIGN * deg)
               + ax.transData)
        for art in self.att_parts[name]:
            art.set_transform(rot)
        self.att_angle[name].set_text(f"{deg:5.1f}°")

    # ══════════════════════════════════════
    #  GUI
    # ══════════════════════════════════════
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Surrogate-MPC Live Simulator")
        self.root.geometry("1500x860")

        # ── control toolbar (tkinter widgets) ──
        bar = tk.Frame(self.root, bd=1, relief="raised", padx=6, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Scenario:").pack(side=tk.LEFT, padx=(2, 2))
        self.var_scenario = tk.StringVar(value="step")
        for label, val in [("Step", "step"), ("Sinusoidal", "sine"), ("Hover", "hover")]:
            tk.Radiobutton(bar, text=label, variable=self.var_scenario, value=val,
                           command=self._on_scenario).pack(side=tk.LEFT)

        tk.Label(bar, text="   Ref:").pack(side=tk.LEFT, padx=(8, 0))
        self.var_ref = tk.IntVar(value=20)
        self.scale_ref = tk.Scale(bar, from_=5, to=30, orient=tk.HORIZONTAL,
                                  length=140, variable=self.var_ref,
                                  command=self._on_ref_live, showvalue=True)
        self.scale_ref.pack(side=tk.LEFT)
        self.scale_ref.bind("<ButtonRelease-1>", self._on_ref_release)

        self.btn_start = tk.Button(bar, text="⏸ Pause", width=9, command=self._toggle_run)
        self.btn_start.pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(bar, text="⚡ Disturb", width=9,
                  command=self._inject_disturbance).pack(side=tk.LEFT, padx=2)

        tk.Label(bar, text="   Speed:").pack(side=tk.LEFT, padx=(8, 0))
        self.var_speed = tk.IntVar(value=1)
        for s in (1, 2, 5):
            tk.Radiobutton(bar, text=f"{s}×", variable=self.var_speed, value=s,
                           command=self._on_speed).pack(side=tk.LEFT)

        tk.Button(bar, text="📷 Export", width=9,
                  command=self._export).pack(side=tk.LEFT, padx=(10, 2))
        self.lbl_status = tk.Label(bar, text="", fg="#444")
        self.lbl_status.pack(side=tk.RIGHT, padx=6)

        # ── figure: [attitude stack | φ ; τ | metrics] ──
        self.fig = Figure(figsize=(15.2, 8.0), dpi=100)
        gs = self.fig.add_gridspec(
            4, 3, width_ratios=[1.0, 2.05, 1.0], height_ratios=[1, 1, 1, 1],
            left=0.015, right=0.985, top=0.95, bottom=0.075, hspace=0.55, wspace=0.12)

        # attitude column (4 rows)
        self.att_parts, self.att_angle, self.att_ax = {}, {}, {}
        for i, name in enumerate(ORDER):
            ax = self.fig.add_subplot(gs[i, 0])
            self._make_attitude_axis(ax, name, COLORS[name])
        # column heading
        self.fig.text(0.025, 0.985, "ATTITUDE FRONT VIEW", fontsize=7.5,
                      fontweight="bold", color="#26D420", ha="left")

        # φ subplot (top of middle column)
        self.ax_phi = self.fig.add_subplot(gs[0:2, 1])
        self.ax_phi.set_title("Roll angle  φ(t)", fontsize=11)
        self.ax_phi.set_ylabel("φ (rad)")
        self.ax_phi.grid(True, alpha=0.3)
        self.line_phi = {
            name: self.ax_phi.plot([], [], color=COLORS[name], lw=2.0, label=name)[0]
            for name in ORDER
        }
        self.line_ref, = self.ax_phi.plot([], [], "--", color=COLORS["ref"], lw=1.6,
                                          label="reference", zorder=5)
        self.ax_phi.legend(loc="upper right", fontsize=8, ncol=3)

        # τ subplot (bottom of middle column)
        self.ax_tau = self.fig.add_subplot(gs[2:4, 1], sharex=self.ax_phi)
        self.ax_tau.set_title("Roll torque  τ_φ(t)", fontsize=11)
        self.ax_tau.set_xlabel("Time (s)")
        self.ax_tau.set_ylabel("τ_φ (mN·m)")
        self.ax_tau.grid(True, alpha=0.3)
        self.line_tau = {
            name: self.ax_tau.plot([], [], color=COLORS[name], lw=1.8, label=name)[0]
            for name in ORDER
        }
        self.ax_tau.axhline(10.0, color="#999", lw=0.8, ls=":")    # ±10 mN·m authority
        self.ax_tau.axhline(-10.0, color="#999", lw=0.8, ls=":")
        self.ax_tau.set_ylim(-12, 12)

        # metrics column
        self.ax_metrics = self.fig.add_subplot(gs[0:4, 2])
        self.ax_metrics.axis("off")
        self.metric_text = {}
        y0, dy = 0.965, 0.245
        for i, name in enumerate(ORDER):
            self.metric_text[name] = self.ax_metrics.text(
                0.02, y0 - i * dy, "", transform=self.ax_metrics.transAxes,
                family="monospace", fontsize=11, va="top", ha="left",
                color=COLORS[name])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.ani = FuncAnimation(self.fig, self._update, interval=30,
                                 blit=False, cache_frame_data=False)

    # ── control callbacks ──
    def _on_scenario(self):
        self.scenario = self.var_scenario.get()
        self.reset_states()

    def _on_ref_live(self, _val):
        self.ref_deg = float(self.var_ref.get())

    def _on_ref_release(self, _evt):
        self.ref_deg = float(self.var_ref.get())
        self.reset_states()

    def _on_speed(self):
        self.speed = int(self.var_speed.get())

    def _toggle_run(self):
        self.running = not self.running
        self.btn_start.config(text="⏸ Pause" if self.running else "▶ Start")

    def _inject_disturbance(self):
        for name in ORDER:
            self.x[name][3] += 2.0                       # p += 2.0 rad/s on all four
        self.lbl_status.config(text="⚡ disturbance injected (p += 2.0 rad/s)")

    def _export(self):
        os.makedirs("plots", exist_ok=True)
        path = os.path.join("plots", "defense_snapshot.png")
        self.fig.savefig(path, dpi=300, bbox_inches="tight")
        self.lbl_status.config(text=f"saved {path} (300 DPI)")
        print(f"Saved snapshot: {path}")

    def _on_close(self):
        try:
            self.ani.event_source.stop()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()

    # ── animation tick ──
    def _update(self, _frame):
        if self.running:
            for _ in range(self.speed):                  # `speed` plant steps per frame
                self.sim_substep()

        # attitude views always reflect current state (even while paused)
        for name in ORDER:
            self._update_attitude(name, float(self.x[name][0]))

        if len(self.t_hist) == 0:
            return []

        t = np.asarray(self.t_hist)
        ref = np.asarray(self.ref_hist)

        x_right = max(WINDOW_S, self.sim_t)
        x_left = x_right - WINDOW_S
        self.ax_phi.set_xlim(x_left, x_right)

        phi_lo, phi_hi = ref.min(), ref.max()
        for name in ORDER:
            phi = np.asarray(self.phi_hist[name])
            self.line_phi[name].set_data(t, phi)
            if phi.size:
                phi_lo = min(phi_lo, phi.min())
                phi_hi = max(phi_hi, phi.max())
            self.line_tau[name].set_data(t, np.asarray(self.tau_hist[name]))
        self.line_ref.set_data(t, ref)

        if phi_hi - phi_lo < 0.05:
            c = 0.5 * (phi_hi + phi_lo)
            phi_lo, phi_hi = c - 0.05, c + 0.05
        pad = 0.12 * (phi_hi - phi_lo)
        self.ax_phi.set_ylim(phi_lo - pad, phi_hi + pad)

        for name in ORDER:
            rmse, settle, over = compute_metrics(self.phi_hist[name], self.ref_hist)
            solve = float(np.mean(self.solve_ms[name])) if self.solve_ms[name] else 0.0
            self.metric_text[name].set_text(
                f"{name}\n"
                f"  RMSE:      {rmse:7.4f} rad\n"
                f"  Settling:  {settle:6.2f} s\n"
                f"  Overshoot: {over:6.1f} %\n"
                f"  Solve:     {solve:6.2f} ms"
            )

        artists = (list(self.line_phi.values()) + [self.line_ref]
                   + list(self.line_tau.values()) + list(self.metric_text.values()))
        for name in ORDER:
            artists += self.att_parts[name] + [self.att_angle[name]]
        return artists

    # ── run ──
    def run(self):
        print("\nLive simulator ready. Controls are in the top toolbar.")
        print("  Scenario: Step / Sinusoidal / Hover   (resets on change)")
        print("  Ref slider 5°–30°                      (resets on release)")
        print("  ⚡ Disturb: p += 2.0 rad/s on all four states")
        print("  Speed 1× / 2× / 5×  |  📷 Export -> plots/defense_snapshot.png (300 DPI)\n")
        self.root.mainloop()


if __name__ == "__main__":
    LiveSimulator().run()