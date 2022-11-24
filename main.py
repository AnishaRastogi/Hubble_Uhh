import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
#np.random.seed(123)
m_true = 50.0
b_true = 0.56
f_true = 0.454

d, v_o = np.genfromtxt('hubble_data - Sheet1.csv', delimiter=',', unpack=True)

print(d)
print(v_o)

def err(distance, velocity):
    h = []
    l = distance.size
    sum_d = np.sum(distance)
    mean_d = sum_d/l
    sum_v = np.sum(velocity)
    mean_v = sum_v/l

    h =[]
    mean_h = []
    sigma_d = []
    sigma_v = []
    simga_h = []
    print(l)
    for i in range(l):
        x = distance[i] - mean_d
        x = abs(x)
        sigma_d.append(x)
        y = velocity[i] - mean_v
        y = abs(y)
        sigma_v.append(y)
        a = distance[i] / velocity[i]
        h.append(a)
    mean_h = np.sum(h)/l
    for i in range(l):
        z = h[i] - mean_h
        print(z)
        z = abs(z)
        print(z)
        sigma_h.append(z)
    return sigma_d, sigma_v, sigma_h
sigma_d = []
sigma_v = []
sigma_h = []
sigma_d, sigma_v, sigma_h = err(d, v_o)
print (sigma_h)
plt.errorbar(d, v_o, yerr= sigma_v, xerr = sigma_d, fmt=".k", capsize=0)
plt.xlabel("distance")
plt.ylabel("Velocity");
def log_likelihood(theta, dist, vel, yerr):
   m, b, log_f = theta
   model = m * dist + vel
   sigma2 = np.power(yerr,2) + np.power(model,2) * np.exp(2 * log_f)
   return -0.5 * np.sum((vel - model) ** 2 / sigma2 + np.log(sigma2))
from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(d, v_o, sigma_v))
m_ml, b_ml, log_f_ml = soln.x
def log_prior(theta):
    m, b, log_f = theta
    if 40.0 < m < 100.0 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf
def log_probability(theta, dist, vel, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dist, vel, sigma_v)

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(d, v_o, sigma_v)
)
sampler.run_mcmc(pos, 5000, progress=True);
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()
tau = sampler.get_autocorr_time()
print(tau)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
);
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(d, np.dot(np.vander(d, 2), sample[:2]), "C1", alpha=0.1)
plt.errorbar(d, v_o, yerr=sigma_v, fmt=".k", capsize=0)
plt.plot(d, m_true * d + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlabel("distance")
plt.ylabel("Velocity");
from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print(display(Math(txt)))
