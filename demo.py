"""Synthetic torsional oscillator; no KUKA data or implementation imports."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def unit_acceleration(t, omega, zeta):
    """Angular acceleration for theta(0)=1 rad, velocity(0)=0."""
    t = np.asarray(t)
    solution = solve_ivp(
        lambda _, x: [x[1], -omega**2*x[0] - 2*zeta*omega*x[1]],
        (0, float(t[-1])), [1., 0.], t_eval=t,
        rtol=1e-9, atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return -omega**2*solution.y[0] - 2*zeta*omega*solution.y[1]


def basis(t, omega, zeta):
    """Exact underdamped response; independently checked against the ODE."""
    wd = omega*np.sqrt(1-zeta*zeta)
    decay = np.exp(-zeta*omega*t)
    theta = decay*(np.cos(wd*t) + zeta*omega/wd*np.sin(wd*t))
    velocity = -omega**2/wd*decay*np.sin(wd*t)
    return -omega**2*theta - 2*zeta*omega*velocity


def fit_physics(t, y):
    """Fit omega, damping and projected amplitude using training values only."""
    def residual(p):
        return p[2]*basis(t, p[0], p[1])-y
    # Predeclared multiple starts; all ranked using TRAINING SSE only.
    fits = [least_squares(residual, [w, .03, .04],
                         bounds=([3., .001, .001], [9., .2, .2]))
            for w in (4., 6., 8.)]
    fit = min(fits, key=lambda f: np.dot(f.fun, f.fun))
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def fit_ar(y, order=8):
    """Fixed-order AR with intercept; no test-dependent order selection."""
    x = np.array([[1., *y[i-order:i][::-1]] for i in range(order, len(y))])
    return np.linalg.lstsq(x, y[order:], rcond=None)[0]


def forecast_ar(history, coefficients, steps):
    values = list(history)
    p = len(coefficients)-1
    for _ in range(steps):
        values.append(float(coefficients[0] + coefficients[1:] @ np.array(values[-p:][::-1])))
    return np.array(values[-steps:])


def experiment(seed=2026):
    t = np.arange(600)/100.  # 100 Hz; training 0--3.99 s; forecast 4--5.99 s
    split = 400
    omega, zeta, radius, amplitude, angle = 6., .03, .5, .1, np.pi/6
    beta = radius*amplitude*np.cos(angle)
    clean = beta*unit_acceleration(t, omega, zeta)
    noise = np.random.default_rng(seed).normal(0, .05, len(t))
    y = clean+noise
    parameters = fit_physics(t[:split], y[:split])
    physical = parameters[2]*basis(t, *parameters[:2])
    ar = forecast_ar(y[:split], fit_ar(y[:split]), len(t)-split)
    predictions = {"physical": physical[split:], "AR8": ar,
                   "zero": np.zeros(len(ar)), "persistence": np.full(len(ar), y[split-1])}
    rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    # Sensor remount at test boundary: same angular motion, different projection.
    shifted = clean[split:]*np.cos(np.pi/3)/np.cos(angle)+noise[split:]
    scores = {name: {"stationary_rmse": rmse(y[split:], pred),
                     "orientation_shift_rmse": rmse(shifted, pred)}
              for name, pred in predictions.items()}
    result = {"seed": seed, "truth": {"omega_rad_s": omega, "zeta": zeta,
              "beta_m": beta, "noise_sd_m_s2": .05},
              "estimate": dict(zip(["omega_rad_s", "zeta", "beta_m"], map(float, parameters))),
              "scores": scores,
              "assumed_orientation_amplitude_rad": {
                  str(deg): float(parameters[2]/(radius*np.cos(np.deg2rad(deg))))
                  for deg in (0, 30, 60)},
              "ode_analytic_max_error": float(np.max(np.abs(clean-beta*basis(t, omega, zeta))))}
    return result, (t[split:]-t[split], y[split:], physical[split:], ar)


def plot_svg(path, arrays):
    """Small dependency-free static plot; identical axes for all forecasts."""
    t, observed, physical, ar = arrays
    ymax = max(float(np.max(np.abs(a))) for a in arrays[1:])*1.1
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390">',
             '<rect width="900" height="390" fill="white"/>',
             '<g font-family="sans-serif" fill="#172b4d"><text x="55" y="25" font-size="19">Synthetic held-out acceleration: fixed-origin 2-second forecast</text>',
             '<text x="360" y="375">Forecast horizon (s)</text>',
             '<text x="12" y="65">m/s²</text></g>']
    for v in (-ymax, 0., ymax):
        yp = 195-130*v/ymax
        parts.append(f'<path d="M65 {yp} H860" stroke="#ddd"/><text x="8" y="{yp+4}" font-size="12">{v:.2f}</text>')
    for v in (0, .5, 1, 1.5, 2):
        xp = 65+795*v/2
        parts.append(f'<text x="{xp}" y="345" font-size="12">{v}</text>')
    for index, (values, color, label) in enumerate(zip(arrays[1:], ('#aaa', '#1464b4', '#c04620'), ('Noisy observations', 'Physical', 'AR8'))):
        points = ' '.join(f'{65+795*x/2:.2f},{195-130*y/ymax:.2f}' for x, y in zip(t, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/><text x="{95+index*240}" y="48" fill="{color}" font-size="14">{label}</text>')
    path.write_text(''.join(parts)+'</svg>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary, arrays = experiment()
    # Fixed seeds; no stopping or selection based on favourable scores.
    repetitions = [experiment(seed)[0] for seed in range(20)]
    result = {"status": "SYNTHETIC ONLY; not empirical KUKA validation",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "primary": primary, "noise_repetitions": repetitions,
              "monte_carlo_note": "20 independent noise draws for one fixed mechanism; not confidence intervals or independent robot experiments."}
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    plot_svg(args.output/'forecast.svg', arrays)
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
