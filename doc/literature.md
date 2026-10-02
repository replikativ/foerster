# Literature

The systems and papers foerster builds on.

## Probabilistic programming systems

- **Anglican** — D. Tolpin, J.-W. van de Meent, H. Yang, F. Wood. *Design and
  Implementation of Probabilistic Programming Language Anglican.* IFL 2016.
  [arXiv:1608.05263](https://arxiv.org/abs/1608.05263) ·
  [github.com/probprog/anglican](https://github.com/probprog/anglican)
- **Daphne** — the probabilistic programming compiler of
  [plai-group/daphne](https://github.com/plai-group/daphne), following
  J.-W. van de Meent, B. Paige, H. Yang, F. Wood. *An Introduction to
  Probabilistic Programming.* [arXiv:1809.10756](https://arxiv.org/abs/1809.10756).
- **Gen** — M. Cusumano-Towner, F. Saad, A. Lew, V. Mansinghka. *Gen: A
  General-Purpose Probabilistic Programming System with Programmable
  Inference.* PLDI 2019. [gen.dev](https://www.gen.dev/)
- **Church** — N. Goodman, V. Mansinghka, D. Roy, K. Bonawitz, J. Tenenbaum.
  *Church: a language for generative models.* UAI 2008.
- C. Weilbach. *Structured Amortized Variational Inference.* PhD thesis,
  University of British Columbia, 2025 — structured continuous normalizing
  flows, graphically structured diffusion models, and Daphne.

## Inference algorithms

- **SMC** — A. Doucet, A. M. Johansen. *A tutorial on particle filtering and
  smoothing: fifteen years later.* 2009.
- **Resampling schemes** — R. Douc, O. Cappé, E. Moulines. *Comparison of
  resampling schemes for particle filtering.* ISPA 2005.
- **Resample-move** — W. R. Gilks, C. Berzuini. *Following a moving target —
  Monte Carlo inference for dynamic Bayesian models.* JRSS B 2001.
- **IBIS** — N. Chopin. *A sequential particle filter method for static
  models.* Biometrika 2002.
- **SMC samplers** (tempered SMC) — P. Del Moral, A. Doucet, A. Jasra.
  *Sequential Monte Carlo samplers.* JRSS B 2006.
- **Adaptive tempering** — Y. Zhou, A. M. Johansen, J. A. D. Aston. *Toward
  automatic model comparison: an adaptive sequential Monte Carlo approach.*
  JCGS 2016.
- **Waste-free SMC** — H.-D. Dau, N. Chopin. *Waste-free sequential Monte
  Carlo.* JRSS B 2022.
- **SMCP3** — A. K. Lew, G. Matheos, T. Zhi-Xuan, M. Ghavamizadeh,
  N. Gothoskar, S. Russell, V. K. Mansinghka. *SMCP3: Sequential Monte Carlo
  with Probabilistic Program Proposals.* AISTATS 2023.
- **Particle cascade** — B. Paige, F. Wood, A. Doucet, Y. W. Teh.
  *Asynchronous Anytime Sequential Monte Carlo.* NeurIPS 2014.
  [arXiv:1407.2864](https://arxiv.org/abs/1407.2864) — the stabilised
  branching rule (Eq. 14) `foerster.cascade` uses.
- **Anytime Monte Carlo** — L. M. Murray, S. S. Singh, A. Lee. *Anytime Monte
  Carlo.* Data-Centric Engineering 2021 — why a run must end by count, not
  by wall clock.
- **Delayed sampling** — L. M. Murray, D. Lundén, J. Kudlicka, D. Broman,
  T. B. Schön. *Delayed Sampling and Automatic Rao-Blackwellization of
  Probabilistic Programs.* AISTATS 2018.
- **Particle MCMC** (PIMH, PMMH, particle Gibbs) — C. Andrieu, A. Doucet,
  R. Holenstein. *Particle Markov chain Monte Carlo methods.* JRSS B 2010.
- **SMC²** — N. Chopin, P. E. Jacob, O. Papaspiliopoulos. *SMC²: an efficient
  algorithm for sequential analysis of state space models.* JRSS B 2013.
- **PGAS** — F. Lindsten, M. I. Jordan, T. B. Schön. *Particle Gibbs with
  ancestor sampling.* JMLR 2014.
- **IPMCMC** — T. Rainforth, C. A. Naesseth, F. Lindsten, B. Paige,
  J.-W. van de Meent, A. Doucet, F. Wood. *Interacting Particle Markov Chain
  Monte Carlo.* ICML 2016.
- **Lightweight MH** — D. Wingate, A. Stuhlmüller, N. Goodman. *Lightweight
  implementations of probabilistic programming languages via transformational
  compilation.* AISTATS 2011.
- **Cycles and mixtures of kernels** — L. Tierney. *Markov chains for
  exploring posterior distributions.* Annals of Statistics 1994.
- **HMC** — R. M. Neal. *MCMC using Hamiltonian dynamics.* Handbook of
  Markov Chain Monte Carlo, 2011.
- **Involutive MCMC** — M. Cusumano-Towner, A. K. Lew, V. Mansinghka.
  *Automating Involutive MCMC using Probabilistic and Differentiable
  Programming.* 2020. [arXiv:2007.09871](https://arxiv.org/abs/2007.09871)
- **BBVI** — R. Ranganath, S. Gerrish, D. Blei. *Black Box Variational
  Inference.* AISTATS 2014.

## Steering and twisted SMC

- **Twisted particle filters** — N. Whiteley, A. Lee. *Twisted particle
  filters.* Annals of Statistics 2014.
- **Twisted SMC for language models** — S. Zhao, R. Brekelmans, A. Makhzani,
  R. Grosse. *Probabilistic Inference in Language Models via Twisted
  Sequential Monte Carlo.* ICML 2024.
  [arXiv:2404.17546](https://arxiv.org/abs/2404.17546)
- **SMC steering** — A. K. Lew, T. Zhi-Xuan, G. Grand, V. K. Mansinghka.
  *Sequential Monte Carlo Steering of Large Language Models using
  Probabilistic Programs.* 2023.
  [arXiv:2306.03081](https://arxiv.org/abs/2306.03081)
- **The reward as a tilt** — T. Korbak, E. Perez, C. L. Buckley. *RL with
  KL penalties is better viewed as Bayesian inference.* Findings of EMNLP
  2022.

## Causality

- J. Pearl. *Causality: Models, Reasoning, and Inference.* 2nd ed., 2009 —
  interventions, the three-step counterfactual (abduction, action,
  prediction), the probability of necessity.

## Checking samplers

- **Simulation-based calibration** — S. Talts, M. Betancourt, D. Simpson,
  A. Vehtari, A. Gelman. *Validating Bayesian Inference Algorithms with
  Simulation-Based Calibration.* 2018.
  [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)

## Monte Carlo, and the name

- N. Metropolis, S. Ulam. *The Monte Carlo Method.* JASA 1949.
- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller,
  E. Teller. *Equation of State Calculations by Fast Computing Machines.*
  J. Chem. Phys. 1953.
- G. Dyson. *Turing's Cathedral.* 2012 — the ENIAC and the Monte Carlo runs
  Klára Dán von Neumann programmed.
- H. von Foerster. *Understanding Understanding: Essays on Cybernetics and
  Cognition.* 2003.
- G. Bateson. *Steps to an Ecology of Mind.* 1972.
