# Run this example from NFSampler root dir
""" 
IMPORTS
"""

### System
import os
import sys
from functools import partial
sys.path.append(".") # this is a temp work around, wouldnt load for me otherwise?

### NF Modules
from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.nfmodel.utils import * #PEP style forbids this
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.Sampler import Sampler
from nfsampler.utils.PRNG_keys import initialize_rng_keys

### Linear Algebra
import numpy as np
import jax
import jax.numpy as jnp  # JAX NumPy
from jax.scipy.special import logsumexp
from jax.config import config
config.update("jax_enable_x64", True)

### Evidence
import harmonic as hm
import emcee

### Plotting
# import corner # I want to use GetDist instead
import getdist
from getdist import plots, MCSamples
getdist.chains.print_load_details = False
import matplotlib.pyplot as plt


""" 
EXAMPLE-SPECIFIC SETUP
"""

# Assign plot saving dir
EXAMPLE = "dualmoon"
CHAINS_DIR = os.path.join("examples/chains", EXAMPLE)
PLOT_DIR = os.path.join("examples/plots", EXAMPLE)
if not os.path.exists(CHAINS_DIR):
    os.makedirs(CHAINS_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


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

""" 
EXAMPLE PARAMS
"""

# Ive edited these so the resulting harmonic chains are (100,1000) to match older examples
n_dim = 2
n_chains = 100 # change to 100 for the normal harmonic chain amount (default 30)
n_loop = 3 # I'm interested in increasing this to see results (default 3)
n_local_steps = 1000 #change to 500 for normal harmonic chain amount (default 1000)
n_global_steps = 1000 #change to 500 for normal harmonic chain amount (default 1000)
learning_rate = 0.1
momentum = 0.9
num_epochs = 5
batch_size = 50
stepsize = 0.01

# """ 
# RUN EXAMPLE
# """


# print("Preparing RNG keys")
# rng_key_set = initialize_rng_keys(n_chains, seed=42)

# print("Initializing MCMC model and normalizing flow model.")

# initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

# model = RealNVP(10, n_dim, 64, 1)
# run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

# print("Initializing sampler class")

# nf_sampler = Sampler(
#     n_dim,
#     rng_key_set,
#     model,
#     run_mcmc,
#     dual_moon_pe,
#     d_likelihood=d_dual_moon,
#     n_loop=n_loop,
#     n_local_steps=n_local_steps,
#     n_global_steps=n_global_steps,
#     n_chains=n_chains,
#     n_epochs=num_epochs,
#     n_nf_samples=100,
#     learning_rate=learning_rate,
#     momentum=momentum,
#     batch_size=batch_size,
#     stepsize=stepsize,
# )

# print("Sampling")

# chains, log_prob, nf_samples, local_accs, global_accs, loss_vals = nf_sampler.sample(
#     initial_position
# )

# # log_probp[chains, steps]

# print(
#     "chains shape: ",
#     chains.shape,
#     "local_accs shape: ",
#     local_accs.shape,
#     "global_accs shape: ",
#     global_accs.shape,
# )

# chains = np.array(chains)
# nf_samples = np.array(nf_samples[1])
# loss_vals = np.array(loss_vals)

# # Plot one chain to show the jump
# plt.figure(figsize=(6, 6))
# axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
# plt.sca(axs[0])
# plt.title("2 chains")
# plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
# plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")

# plt.sca(axs[1])
# plt.title("NF loss")
# plt.plot(loss_vals)
# plt.xlabel("iteration")

# plt.sca(axs[2])
# plt.title("Local Acceptance")
# plt.plot(local_accs.mean(0))
# plt.xlabel("iteration")

# plt.sca(axs[3])
# plt.title("Global Acceptance")
# plt.plot(global_accs.mean(0))
# plt.xlabel("iteration")
# plt.tight_layout()
# # plt.show(block=False)
# plt.savefig(os.path.join(PLOT_DIR, "chain_jump.pdf"))

# # Triangle plot
# # Plot all chains
# # figure = corner.corner(
# #     chains.reshape(-1, n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
# # )
xtags=["x%s"%i for i in range(n_dim)]
# ztags=["x%s"%i for i in range(n_dim)]

# chainsMC = MCSamples(samples=chains, names=xtags, labels=xtags, label="Posterior")
# g = plots.get_subplot_plotter()
# g.triangle_plot([chainsMC], filled=True)
# g.export(os.path.join(PLOT_DIR, "posterior_samples.pdf"))
# # # figure.set_size_inches(7, 7)
# # # figure.suptitle("Visualize samples")
# # # plt.show(block=False)
# # # plt.savefig("test.png", dpi=300)
# # plt.savefig(os.path.join(PLOT_DIR, "posterior_samples.png"), dpi=300)

# # Plot Nf samples (base distribution)
# nf_samplesMC = MCSamples(samples=nf_samples, names=ztags, labels=ztags, label="Base")
# g = plots.get_subplot_plotter()
# g.triangle_plot([nf_samplesMC], filled=True)
# g.export(os.path.join(PLOT_DIR, "base_samples.pdf"))
# # figure = corner.corner(nf_samples, labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
# # figure.set_size_inches(7, 7)
# # figure.suptitle("Visualize NF samples")
# # plt.show(block=False)
# # plt.savefig(os.path.join(PLOT_DIR, "base_samples.png"), dpi=300)



# """ 
# RUN HARMONIC FOR EVIDENCE WORK
# """

# # Now need samples and lnprob

# samples = np.array(chains[:, -(n_local_steps + n_global_steps) :, :])
# lnprob = np.array(log_prob[:, -(n_local_steps + n_global_steps) :])

# ### save samples to file for speed
# np.save(os.path.join(CHAINS_DIR, "samples.npy"), samples)
# np.save(os.path.join(CHAINS_DIR, "lnprob.npy"), lnprob)

# print(
#     f"Input Harmonic Samples Shape: {samples.shape}",
#     f"Input Harmonic lnprob shape: {lnprob.shape}"
# )

# # Compute analytic evidence.
# if n_dim == 2:

#     xmin = -3
#     xmax = 3
#     nx = 100

#     x = np.linspace(xmin, xmax, nx)
#     y = np.linspace(xmin, xmax, nx)
#     x_grid, y_grid = np.meshgrid(x, y)
#     ln_posterior_grid = np.zeros((nx, nx))
#     for i in range(nx):
#         for j in range(nx):
#             ln_posterior_grid[i, j] = dual_moon_pe(
#                 np.array([x_grid[i, j], y_grid[i, j]])
#             )

#     dx = x_grid[0, 1] - x_grid[0, 0]
#     dy = y_grid[1, 0] - y_grid[0, 0]
#     evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy

#     print(f"evidence_numerical_integration = {evidence_numerical_integration}")
#     # Im getting 0.6 for evidence rather than 1.8 reported in last weeks' meeting!

samples = np.load(os.path.join(CHAINS_DIR, "samples.npy"))
lnprob = np.load(os.path.join(CHAINS_DIR, "lnprob.npy"))

chains = hm.Chains(n_dim)
chains.add_chains_3d(samples, lnprob)
chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.2)

samplesMC = MCSamples(samples = samples, names = xtags, labels = xtags, label = 'Posterior')

g = plots.get_subplot_plotter()
g.triangle_plot([samplesMC], filled=True)
g.export(os.path.join(PLOT_DIR, "harmonic_samples.pdf")) # they look fine!


print(f"Harmonic Samples Shape: {samples.shape}")

domain_kde = [] # not used for KDE model
kde_diameter = 0.003 # forcing to 0.003 (dont use cross val)
model_kde = hm.model.KernelDensityEstimate(n_dim, domain_kde, hyper_parameters=[kde_diameter])
fit_success_kde = model_kde.fit(chains_train.samples, chains_train.ln_posterior)

ev_kde = hm.Evidence(chains_test.nchains, model_kde)
ev_kde.add_chains(chains_test)
evidence_kde, evidence_std_kde = ev_kde.compute_evidence()
print(f"evidence_numerical_integration = 0.5953164176730806")
print('Harmonic Evidence = {} ± {}'.format(evidence_kde, evidence_std_kde))
print(ev_kde.nsamples_eff_per_chain)

# DONE: Redo CrossVal - once done delete all so it looks simpler
# DONE: print number of effective samples of chains/model
# DONE: Explore training proportion
# DONE: Different KDE radius untul effective samples are a good level
# DONE: Evaluate model on the grid (gives diff answer from analytical and harmonic ev)
# DONE: Plot model image (only works for 2d)
# DONE: Sample from model and getdist plot it (emcee)
# TODO: Sample from model and getdist plot it (nfsampler)
# TODO: clean up logspace discrepency
# TODO: clean up code completely

"""
EVALUATE MODEL ON GRID

All I need is the 2d support grid and a way to evaluate the model at each point
This can be done with kde_model.predict methinks
"""

# NOTE: is it logspace etc? will that be a problem?

# Compute analytic evidence.
if n_dim == 2:

    xmin = -3
    xmax = 3
    nx = 100

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(xmin, xmax, nx)
    x_grid, y_grid = np.meshgrid(x, y)
    ln_posterior_grid = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            ln_posterior_grid[i, j] = np.exp(model_kde.predict(
                np.array([x_grid[i, j], y_grid[i, j]])))

    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]
    evidence_model = np.sum(ln_posterior_grid) * dx * dy

    print(f"evidence_model = {evidence_model}")
    # Getting 1.0 ish for model evaluated on grid, lets plot the image of the model 
    # as its in 2d first. Then sample from it

""" 
PLOT MODEL IN 2D
"""

# NOTE: Need to think about if it's in logspace or not

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.imshow(ln_posterior_grid, cmap='jet', origin='lower', aspect='auto',
                extent=[np.min(x_grid), np.max(x_grid),
                        np.min(y_grid), np.max(y_grid)])
plt.colorbar(im)                        

plt.savefig(os.path.join(PLOT_DIR, "model_image.pdf"))

# """ 
# SAMPLE FROM MODEL IN 2D USING EMCEE
# """

# # Start by using emcee for the meetin today then convert to NFSampler over the weekend

# # set up MCMC params
# nchains_mc = 200
# samples_per_chain = 5000
# nburn_mc = 2000


# # initialise random seed
# np.random.seed(1) # for reproducibility

# # Set initial random position and state
# pos = np.random.rand(nchains_mc,n_dim) * 1. + np.zeros(n_dim)
# rstate = np.random.get_state() # same each time as we have set the seed?

# # Instantiate and execute sampler 
# sampler = emcee.EnsembleSampler(nchains_mc, n_dim, lambda x : model_kde.predict(x), args = [])
# sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate) 
# samples_model = np.ascontiguousarray(sampler.chain[:,nburn_mc:,:])
# # lnprob_model = np.ascontiguousarray(sampler.lnprobability[:,nburn_mc:])

# samples_model_MC = MCSamples(samples = samples_model.reshape(-1,n_dim), names = xtags, labels = xtags, label="Model Emcee")
# g = plots.getSubplotPlotter()
# g.triangle_plot([samples_model_MC], filled=True)
# g.export(os.path.join(PLOT_DIR, "model_samples_emcee.pdf"))

# g = plots.getSubplotPlotter()
# g.triangle_plot([samplesMC, samples_model_MC], filled=False)
# g.export(os.path.join(PLOT_DIR, "compare_samples_emcee.pdf"))

""" 
SAMPLE FROM MODEL IN 2D USING NFSampler
"""

# Ive edited these so the resulting harmonic chains are (100,1000) to match older examples
n_dim = 2
n_chains = 100 # change to 100 for the normal harmonic chain amount (default 30)
n_loop = 3 # I'm interested in increasing this to see results (default 3)
n_local_steps = 1000 #change to 500 for normal harmonic chain amount (default 1000)
n_global_steps = 1000 #change to 500 for normal harmonic chain amount (default 1000)
learning_rate = 0.1
momentum = 0.9
num_epochs = 5
batch_size = 50
stepsize = 0.01

@partial(static_argnums=1)
def model_func(x):
    x_array = np.array(x)
    return model_kde.predict(x_array)

d_model_func = jax.grad(model_func)

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

model = RealNVP(10, n_dim, 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

print("Initializing sampler class")

nf_sampler_model = Sampler(
    n_dim,
    rng_key_set,
    model,
    run_mcmc,
    model_func,
    d_likelihood=d_model_func,
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

chains_model, _,_,_,_,_ = nf_sampler_model.sample(initial_position)



# ### copied in the cython code for the model.predict
# def predict(self, np.ndarray[double, ndim=1, mode="c"] x):
#         """Predict the value of the posterior at point x.

#         Args:

#             x (double ndarray[ndim]): 1D array of sample of shape (ndim) to predict
#                 posterior value.
        
#         Returns:

#             (double): Predicted log_e posterior value.

#         """
#         samples = self.samples        
#         start_end = self.start_end
#         inv_scales = self.inv_scales
#         nsamples = samples.shape[0], 
#         ndim = self.ndim
#         ngrid = self.ngrid
#         cdef double inv_diam = 1.0/self.D, radius_squared = self.radius_squared
#         cdef dict grid = self.grid

#         cdef long count = 0
        
#         # Find the pixel that the sample is in
#         index = 0
#         for i_dim in range(ndim):
#             sub_index = <long>((x[i_dim]-start_end[i_dim,0]) * \
#                 inv_scales[i_dim]*inv_diam) + 1
#             index += sub_index*ngrid**i_dim

#         KernelDensityEstimate_loop_round_and_search(index, i_dim, ngrid, ndim, 
#                                                     grid, samples, x, 
#                                                     inv_scales, radius_squared, 
#                                                     &count)

#         return log(count) - self.ln_norm