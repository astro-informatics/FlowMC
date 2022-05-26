from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
import jax
import jax.numpy as jnp  # JAX NumPy
from nfsampler.sampler.Sampler import Sampler
from nfsampler.utils.PRNG_keys import initialize_rng_keys
from jax.scipy.special import logsumexp
import numpy as np

import harmonic as hm
from jax.config import config

config.update("jax_enable_x64", True)

from nfsampler.nfmodel.utils import *


def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    # term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    # return -(term1 - logsumexp(term2) - logsumexp(term3))
    return -(term1 - logsumexp(term2))


d_dual_moon = jax.grad(dual_moon_pe)

### Demo config

n_dim = 2
n_chains = 30
n_loop = 3
n_local_steps = 1000
n_global_steps = 1000
learning_rate = 0.1
momentum = 0.9
num_epochs = 5
batch_size = 50
stepsize = 0.01

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1


model = RealNVP(10, n_dim, 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    model,
    run_mcmc,
    dual_moon_pe,
    d_likelihood=d_dual_moon,
    n_loop=n_loop,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    n_nf_samples=100,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    stepsize=stepsize,
)

print("Sampling")

chains, log_prob, nf_samples, local_accs, global_accs, loss_vals = nf_sampler.sample(
    initial_position
)

# log_probp[chains, steps]

print(
    "chains shape: ",
    chains.shape,
    "local_accs shape: ",
    local_accs.shape,
    "global_accs shape: ",
    global_accs.shape,
)

chains = np.array(chains)
nf_samples = np.array(nf_samples[1])
loss_vals = np.array(loss_vals)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.figure(figsize=(6, 6))
axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
plt.sca(axs[0])
plt.title("2 chains")
plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.sca(axs[1])
plt.title("NF loss")
plt.plot(loss_vals)
plt.xlabel("iteration")

plt.sca(axs[2])
plt.title("Local Acceptance")
plt.plot(local_accs.mean(0))
plt.xlabel("iteration")

plt.sca(axs[3])
plt.title("Global Acceptance")
plt.plot(global_accs.mean(0))
plt.xlabel("iteration")
plt.tight_layout()
plt.show(block=False)

# Plot all chains
figure = corner.corner(
    chains.reshape(-1, n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
)
figure.set_size_inches(7, 7)
figure.suptitle("Visualize samples")
plt.show(block=False)

# Plot Nf samples
figure = corner.corner(nf_samples, labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
figure.set_size_inches(7, 7)
figure.suptitle("Visualize NF samples")
plt.show(block=False)


# Compute analytic evidence.
if n_dim == 2:

    xmin = -2.5
    xmax = 2.5
    nx = 100

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(xmin, xmax, nx)
    x_grid, y_grid = np.meshgrid(x, y)
    ln_posterior_grid = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            ln_posterior_grid[i, j] = dual_moon_pe(
                np.array([x_grid[i, j], y_grid[i, j]])
            )

    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]
    evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy

    print(f"evidence_numerical_integration = {evidence_numerical_integration}")


# Now need samples and lnprob

samples = np.array(chains[:, -(n_local_steps + n_global_steps) :, :])
lnprob = np.array(log_prob[:, -(n_local_steps + n_global_steps) :])

chains = hm.Chains(n_dim)
chains.add_chains_3d(samples, lnprob)
chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)


domain = []
nhyper = 2
step = -2
nfold = 2
hyper_parameters = [[10 ** (R)] for R in range(-nhyper + step, step)]
validation_variances = hm.utils.cross_validation(
    chains_train,
    domain,
    hyper_parameters,
    nfold=nfold,
    modelClass=hm.model.KernelDensityEstimate,
    seed=0,
)

print(f"validation_variances = {validation_variances}")

best_hyper_param_ind = np.argmin(validation_variances)
best_hyper_param = hyper_parameters[best_hyper_param_ind]
print(f"Best hyper-parameter = {best_hyper_param}")

model = hm.model.KernelDensityEstimate(n_dim, domain, hyper_parameters=best_hyper_param)

fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
print(f"Fit success = {fit_success}")

ev = hm.Evidence(chains_test.nchains, model)
ev.add_chains(chains_test)
ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

np.exp(ln_evidence)
print(f"evidence_harmonic = {np.exp(ln_evidence)}")
