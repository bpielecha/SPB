import numpy as np
import matplotlib.pyplot as plt

PARAMS = {
    'p1': 8.8,
    'p2': 440,
    'p3': 100,
    'd1': 1.375e-14,
    'd2': 1.375e-4,
    'd3': 3e-5,
    'k1': 1.925e-4,
    'k2': 1e5,
    'k3': 1.5e5
}

MODS = {
    'sirna': 0.02,
    'dna': 0.1
}

def model(s, t, cfg, p):
    x, y, z, w = s
    d_ok = cfg['dna_ok']
    s_on = cfg['siRNA']
    w_on = cfg['pten_active']

    dx = p['p1'] - p['d1'] * x * z**2
    m1 = MODS['sirna'] if s_on else 1.0
    m2 = MODS['dna'] if d_ok else 1.0

    y_prod = p['p2'] * m1 * (x**4) / ((x**4) + p['k2']**4)
    y_tr = p['k1'] * p['k3']**2 / (p['k3']**2 + w**2) * y
    y_deg = p['d2'] * m2 * y
    dy = y_prod - y_tr - y_deg

    z_deg = p['d2'] * m2 * z
    dz = y_tr - z_deg

    w_gen = p['p3'] * (x**4) / ((x**4) + p['k2']**4) if w_on else 0.0
    dw = w_gen - p['d3'] * w

    return np.array([dx, dy, dz, dw])

def rk4(f, y0, t_arr, args):
    res = np.zeros((len(t_arr), len(y0)))
    res[0] = y0
    for i in range(1, len(t_arr)):
        h = t_arr[i] - t_arr[i - 1]
        t = t_arr[i - 1]
        y = res[i - 1]
        k1 = f(y, t, *args)
        k2 = f(y + 0.5 * h * k1, t + 0.5 * h, *args)
        k3 = f(y + 0.5 * h * k2, t + 0.5 * h, *args)
        k4 = f(y + h * k3, t + h, *args)
        res[i] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return res

def simulate(y0, t_arr, cfg, p):
    return rk4(model, y0, t_arr, (cfg, p))

def local_sens(pn, y0, t_arr, p_base, cfg, delta=0.01):
    p_up = p_base.copy()
    p_dn = p_base.copy()
    p_up[pn] *= (1 + delta)
    p_dn[pn] *= (1 - delta)

    up = simulate(y0, t_arr, cfg, p_up)[:, 0]
    dn = simulate(y0, t_arr, cfg, p_dn)[:, 0]

    return (up - dn) / (2 * delta * p_base[pn])

def global_sens(y0, t_arr, p_base, cfg, trials=500):
    vals = []
    sets = []
    for _ in range(trials):
        s = {k: np.random.uniform(0.8 * v, 1.2 * v) for k, v in p_base.items()}
        sim = simulate(y0, t_arr, cfg, s)
        vals.append(sim[-1, 0])
        sets.append(s)

    vals = np.array(vals)
    res = {}
    for k in p_base:
        samp = np.array([s[k] for s in sets])
        res[k] = np.corrcoef(samp, vals)[0, 1] ** 2 if np.std(samp) > 0 and np.std(vals) > 0 else 0
    return res

def global_time_series(y0, t_arr, p_base, cfg, trials=100):
    n = len(t_arr)
    data = np.zeros((trials, n))
    samples = []

    for i in range(trials):
        s = {k: np.random.uniform(0.8 * v, 1.2 * v) for k, v in p_base.items()}
        sim = simulate(y0, t_arr, cfg, s)
        data[i] = sim[:, 0]
        samples.append(s)

    out = {k: [] for k in p_base}
    for j in range(n):
        v = data[:, j]
        for k in p_base:
            s_vals = np.array([s[k] for s in samples])
            score = np.corrcoef(s_vals, v)[0, 1] ** 2 if np.std(s_vals) > 0 and np.std(v) > 0 else 0
            out[k].append(score)
    return out

def param_response(pn, y0, t_arr, p_base, cfg):
    out = {}
    for s in [0.8, 1.0, 1.2]:
        p = p_base.copy()
        p[pn] *= s
        out[s] = simulate(y0, t_arr, cfg, p)[:, 0]
    return out

def plot_curve(t, data, title, path=None):
    plt.figure()
    for label, y in data.items():
        plt.plot(t, y, label=label)
    plt.title(title)
    plt.xlabel("Time [min]")
    plt.ylabel("Relative sensitivity (p53)")
    plt.legend(title="Parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_variation(t, data, pn, scen, mode, tag, path=None):
    plt.figure()
    for s, y in data.items():
        label = "baseline" if np.isclose(s, 1.0) else f"{int(s*100)}% of baseline"
        plt.plot(t, y, label=label, linestyle='--' if s != 1.0 else '-')
    plt.title(f"Effect of '{pn}' on p53\nScenario: {scen}, mode: {mode}")
    plt.xlabel("Time [min]")
    plt.ylabel("p53 level")
    plt.legend(title="Parameter variant")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    t = np.linspace(0, 48*60, 1000)
    y0 = [20000, 120000, 30000, 40000]

    CASES = {
        "normal": {'siRNA': False, 'pten_active': True, 'dna_ok': True},
        "cancer": {'siRNA': False, 'pten_active': False, 'dna_ok': False}
    }

    for scen, cfg in CASES.items():
        print(f"\n--- Scenario: {scen} ---")
        avg_imp = {}
        end_imp = {}

        for p in PARAMS:
            s = local_sens(p, y0, t, PARAMS, cfg)
            avg_imp[p] = np.mean(np.abs(s))
            end_imp[p] = np.abs(s[-1])

        print("\nLocal (mean):")
        for k, v in sorted(avg_imp.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.2e}")

        print("\nLocal (end at 48h):")
        for k, v in sorted(end_imp.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.2e}")

        g = global_sens(y0, t, PARAMS, cfg)
        print("\nGlobal (Sobol-like):")
        for k, v in sorted(g.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.4f}")

        top_l = max(avg_imp, key=avg_imp.get)
        bot_l = min(avg_imp, key=avg_imp.get)
        sens = {
            f"Top ({top_l})": local_sens(top_l, y0, t, PARAMS, cfg),
            f"Bottom ({bot_l})": local_sens(bot_l, y0, t, PARAMS, cfg)
        }
        plot_curve(t, sens, f"Sensitivity functions - {scen}")

        for p in [top_l, bot_l]:
            v = param_response(p, y0, t, PARAMS, cfg)
            plot_variation(t, v, p, scen, "local", "change")

        top_g = max(g, key=g.get)
        bot_g = min(g, key=g.get)

        for p in [top_g, bot_g]:
            v = param_response(p, y0, t, PARAMS, cfg)
            plot_variation(t, v, p, scen, "global", "change")

        ts = global_time_series(y0, t, PARAMS, cfg)
        plot_curve(
            t,
            {
                f"Top (global {top_g})": ts[top_g],
                f"Bottom (global {bot_g})": ts[bot_g]
            },
            f"Global sensitivity (Sobol-like) - {scen}"
        )
