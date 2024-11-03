#!/usr/bin/env python3
"""
File: multi_round_inference.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: 
"""
import sys
import pickle
import torch
import numpy as np
from torch import tensor
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE
from sbi import analysis
from LFI_extragalactic import LFI_EG

param_file, num_rounds = sys.argv[1:]

elefeye = LFI_EG(param_file)

simulator = elefeye.simulator
prior = elefeye.priors

inference = SNPE(prior)

observation_params = tensor([-7, 2])
x_0 = elefeye.simulator(observation_params)

for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000)

    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()

    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)


print(f'Posterior estimated after {num_rounds} rounds')

with open(elefeye.output_path + "posteriors.pickle", 'wb') as f:
    pickle.dump(posteriors, f)


posterior = posteriors[-1]
posterior_samples = posterior.sample((50000,), x=x_0)
fig, ax = analysis.pairplot(posterior_samples)

fig.savefig(elefeye.output_path + "pairplot.png")
