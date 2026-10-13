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
- **WebPPL** — N. D. Goodman, A. Stuhlmüller. *The Design and
  Implementation of Probabilistic Programming Languages.* 2014.
  [dippl.org](https://dippl.org) · [webppl.org](https://webppl.org) —
  JavaScript with `sample`, `factor` and `Infer`, implemented by CPS
  transformation. Its sources include an asynchronous particle filter
  (`asyncpf`) and particle MCMC (`pmcmc`): prior art for foerster's
  particle cascade and particle-MCMC methods.
- **Stan** — B. Carpenter, A. Gelman, M. D. Hoffman, D. Lee, B. Goodrich,
  M. Betancourt, M. Brubaker, J. Guo, P. Li, A. Riddell. *Stan: A
  Probabilistic Programming Language.* Journal of Statistical Software 76(1),
  2017.
- **PyMC** — O. Abril-Pla, V. Andreani, C. Carroll, L. Dong,
  C. J. Fonnesbeck, M. Kochurov, R. Kumar, J. Lao, C. C. Luhmann,
  O. A. Martin, M. Osthege, R. Vieira, T. Wiecki, R. Zinkov. *PyMC: a
  modern, and comprehensive probabilistic programming framework in Python.*
  PeerJ Computer Science 9:e1516, 2023.
- **Pyro** — E. Bingham, J. P. Chen, M. Jankowiak, F. Obermeyer,
  N. Pradhan, T. Karaletsos, R. Singh, P. Szerlip, P. Horsfall,
  N. D. Goodman. *Pyro: Deep Universal Probabilistic Programming.* JMLR
  20(28), 2019.
- **NumPyro** — D. Phan, N. Pradhan, M. Jankowiak. *Composable Effects for
  Flexible and Accelerated Probabilistic Programming in NumPyro.* 2019.
  [arXiv:1912.11554](https://arxiv.org/abs/1912.11554)
- **Turing** — H. Ge, K. Xu, Z. Ghahramani. *Turing: A Language for
  Flexible Probabilistic Inference.* AISTATS 2018.
- C. Weilbach. *Structured Amortized Variational Inference.* PhD thesis,
  University of British Columbia, 2025 — structured continuous normalizing
  flows, graphically structured diffusion models, and Daphne.

## Inference algorithms

Each entry names the foerster function that implements it. The
[inference methods notebook](https://replikativ.github.io/foerster/foerster.methods.html)
runs them side by side.

### Importance sampling and SMC

- **SMC** (`smc-infer`) — A. Doucet, A. M. Johansen. *A tutorial on
  particle filtering and smoothing: fifteen years later.* In The Oxford
  Handbook of Nonlinear Filtering, 2011 (tech. report 2008).
- **Resampling schemes** (`:resampling`) — R. Douc, O. Cappé, E. Moulines.
  *Comparison of resampling schemes for particle filtering.* ISPA 2005.
  [arXiv:cs/0507025](https://arxiv.org/abs/cs/0507025)
- **Resample-move** (`:anchors`, `:rejuvenate`) — W. R. Gilks, C. Berzuini.
  *Following a moving target — Monte Carlo inference for dynamic Bayesian
  models.* JRSS B 63(1):127–146, 2001.
  [doi:10.1111/1467-9868.00280](https://doi.org/10.1111/1467-9868.00280)
- **IBIS** (resample-move on static parameters) — N. Chopin. *A sequential
  particle filter method for static models.* Biometrika 89(3):539–552, 2002.
  [doi:10.1093/biomet/89.3.539](https://doi.org/10.1093/biomet/89.3.539)
- **SMC samplers** (`tempered-infer`) — P. Del Moral, A. Doucet, A. Jasra.
  *Sequential Monte Carlo samplers.* JRSS B 68(3):411–436, 2006.
  [doi:10.1111/j.1467-9868.2006.00553.x](https://doi.org/10.1111/j.1467-9868.2006.00553.x)
- **Adaptive tempering** (the conditional-ESS schedule) — Y. Zhou,
  A. M. Johansen, J. A. D. Aston. *Toward automatic model comparison: an
  adaptive sequential Monte Carlo approach.* JCGS 25(3):701–726, 2016.
  [doi:10.1080/10618600.2015.1060885](https://doi.org/10.1080/10618600.2015.1060885)
- **Waste-free SMC** (`:waste-free`) — H.-D. Dau, N. Chopin. *Waste-free
  sequential Monte Carlo.* JRSS B 84(1):114–148, 2022.
  [arXiv:2011.02328](https://arxiv.org/abs/2011.02328)
- **SMCP3** (`:smcp3`) — A. K. Lew, G. Matheos, T. Zhi-Xuan,
  M. Ghavamizadeh, N. Gothoskar, S. Russell, V. K. Mansinghka. *SMCP3:
  Sequential Monte Carlo with Probabilistic Program Proposals.* AISTATS
  2023, PMLR 206:7061–7088.
  [proceedings.mlr.press/v206/lew23a](https://proceedings.mlr.press/v206/lew23a.html)
- **Particle cascade** (`cascade/cascade`, and the per-batch condition of
  `:batch`) — B. Paige, F. Wood, A. Doucet, Y. W. Teh. *Asynchronous Anytime
  Sequential Monte Carlo.* NeurIPS 2014.
  [arXiv:1407.2864](https://arxiv.org/abs/1407.2864) — the stabilised
  branching rule (Eq. 14) `foerster.cascade` uses.
- **Anytime Monte Carlo** — L. M. Murray, S. S. Singh, A. Lee. *Anytime Monte
  Carlo.* Data-Centric Engineering 2:e27, 2021.
  [arXiv:1612.03319](https://arxiv.org/abs/1612.03319) — why a run must end
  by count, not by wall clock.
- **Delayed sampling** (`foerster.conjugate`) — L. M. Murray, D. Lundén,
  J. Kudlicka, D. Broman, T. B. Schön. *Delayed Sampling and Automatic
  Rao-Blackwellization of Probabilistic Programs.* AISTATS 2018.
  [arXiv:1708.07787](https://arxiv.org/abs/1708.07787)
- **Sequential core-set Monte Carlo** (`scmc/scmc`) — B. Beronov,
  C. Weilbach, F. Wood, T. Campbell. *Sequential core-set Monte Carlo.* UAI
  2021, PMLR 161:2165–2175.
  [proceedings.mlr.press/v161/beronov21a](https://proceedings.mlr.press/v161/beronov21a.html)
- **GIGA** (`scmc/giga`) — T. Campbell, T. Broderick. *Bayesian Coreset
  Construction via Greedy Iterative Geodesic Ascent.* ICML 2018.
  [arXiv:1802.01737](https://arxiv.org/abs/1802.01737) — and the Hilbert
  coreset framework it belongs to: T. Campbell, T. Broderick. *Automated
  Scalable Bayesian Inference via Hilbert Coresets.* JMLR 20(15), 2019.
  [arXiv:1710.05053](https://arxiv.org/abs/1710.05053)

### Particle MCMC

- **PIMH, particle Gibbs, PMMH** (`pimh-infer`, `pgibbs-infer`,
  `smc2/pmmh`) — C. Andrieu, A. Doucet, R. Holenstein. *Particle Markov
  chain Monte Carlo methods.* JRSS B 72(3):269–342, 2010.
  [doi:10.1111/j.1467-9868.2009.00736.x](https://doi.org/10.1111/j.1467-9868.2009.00736.x)
- **PGAS** (`pgas-infer`) — F. Lindsten, M. I. Jordan, T. B. Schön.
  *Particle Gibbs with ancestor sampling.* JMLR 15:2145–2184, 2014.
  [arXiv:1401.0604](https://arxiv.org/abs/1401.0604)
- **IPMCMC** (`ipmcmc-infer`) — T. Rainforth, C. A. Naesseth, F. Lindsten,
  B. Paige, J.-W. van de Meent, A. Doucet, F. Wood. *Interacting Particle
  Markov Chain Monte Carlo.* ICML 2016.
  [arXiv:1602.05128](https://arxiv.org/abs/1602.05128)
- **SMC²** (`smc2/smc2`) — N. Chopin, P. E. Jacob, O. Papaspiliopoulos.
  *SMC²: an efficient algorithm for sequential analysis of state space
  models.* JRSS B 75(3):397–426, 2013.
  [arXiv:1101.1528](https://arxiv.org/abs/1101.1528)

### Markov chains over traces

- **Metropolis–Hastings** — N. Metropolis, A. W. Rosenbluth,
  M. N. Rosenbluth, A. H. Teller, E. Teller (1953, below); W. K. Hastings.
  *Monte Carlo sampling methods using Markov chains and their
  applications.* Biometrika 57(1):97–109, 1970.
  [doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
- **Lightweight MH** (`:mh`, single-site MH over addressed traces) —
  D. Wingate, A. Stuhlmüller, N. Goodman. *Lightweight implementations of
  probabilistic programming languages via transformational compilation.*
  AISTATS 2011, PMLR 15:770–778.
- **Cycles and mixtures of kernels** (`k/cycle`, `k/mixture`) — L. Tierney.
  *Markov chains for exploring posterior distributions.* Annals of
  Statistics 22(4):1701–1728, 1994.
  [doi:10.1214/aos/1176325750](https://doi.org/10.1214/aos/1176325750)
- **Involutive MCMC** (`involutive/step`) — K. Neklyudov, M. Welling,
  E. Egorov, D. Vetrov. *Involutive MCMC: a Unifying Framework.* ICML 2020.
  [arXiv:2006.16653](https://arxiv.org/abs/2006.16653); and as Gen automates
  it: M. Cusumano-Towner, A. K. Lew, V. Mansinghka. *Automating Involutive
  MCMC using Probabilistic and Differentiable Programming.* 2020.
  [arXiv:2007.09871](https://arxiv.org/abs/2007.09871)
- **Reversible jump** — P. J. Green. *Reversible jump Markov chain Monte
  Carlo computation and Bayesian model determination.* Biometrika
  82(4):711–732, 1995.
  [doi:10.1093/biomet/82.4.711](https://doi.org/10.1093/biomet/82.4.711)
- **Delayed acceptance** (`k/delayed-acceptance-kernel`) — J. A. Christen,
  C. Fox. *Markov chain Monte Carlo using an approximation.* JCGS
  14(4):795–810, 2005.
  [doi:10.1198/106186005X76983](https://doi.org/10.1198/106186005X76983);
  the two-stage ratio as foerster writes it: M. B. Lykkegaard, T. J. Dodwell,
  C. Fox, G. Mingas, R. Scheichl. *Multilevel Delayed Acceptance MCMC.*
  SIAM/ASA JUQ 11(1):1–30, 2023 (Lemma 2.4).
  [doi:10.1137/22M1476770](https://doi.org/10.1137/22M1476770)

### Hamiltonian Monte Carlo

- **HMC** (`k/hmc-kernel`) — S. Duane, A. D. Kennedy, B. J. Pendleton,
  D. Roweth. *Hybrid Monte Carlo.* Physics Letters B 195(2):216–222, 1987.
  [doi:10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693(87)91197-X);
  R. M. Neal. *MCMC using Hamiltonian dynamics.* Handbook of Markov Chain
  Monte Carlo, 2011. [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)
- **NUTS** (`k/nuts-kernel`, `:nuts`) — M. D. Hoffman, A. Gelman. *The
  No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
  Carlo.* JMLR 15:1593–1623, 2014.
  [arXiv:1111.4246](https://arxiv.org/abs/1111.4246) — with the
  multinomial trajectories and generalized U-turn criterion of
  M. Betancourt. *A Conceptual Introduction to Hamiltonian Monte Carlo.*
  2017. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
- **Dual averaging** (step-size adaptation) — Y. Nesterov. *Primal-dual
  subgradient methods for convex problems.* Mathematical Programming
  120:221–259, 2009.
  [doi:10.1007/s10107-007-0149-x](https://doi.org/10.1007/s10107-007-0149-x)
- **Fisher-divergence metric** (`:metric :fisher`) — A. Seyboldt,
  E. L. Carlson, B. Carpenter. *Preconditioning Hamiltonian Monte Carlo by
  minimizing Fisher Divergence.* 2026.
  [arXiv:2603.18845](https://arxiv.org/abs/2603.18845) — the diagonal of
  nutpie's adaptation.

### Tempering between chains

- **Parallel tempering** — C. J. Geyer. *Markov chain Monte Carlo maximum
  likelihood.* Computing Science and Statistics: Proceedings of the 23rd
  Symposium on the Interface, 1991.
- **Non-reversible parallel tempering** (`parallel-tempering-infer`) —
  S. Syed, A. Bouchard-Côté, G. Deligiannidis, A. Doucet. *Non-reversible
  parallel tempering: a scalable highly parallel MCMC scheme.* JRSS B
  84(2):321–350, 2022.
  [doi:10.1111/rssb.12464](https://doi.org/10.1111/rssb.12464) ·
  [arXiv:1905.02939](https://arxiv.org/abs/1905.02939) — the deterministic
  even–odd swaps and the schedule tuned in the communication barrier.
- **Stepping-stone evidence** (PT's `m/log-marginal`) — W. Xie, P. O. Lewis,
  Y. Fan, L. Kuo, M.-H. Chen. *Improving marginal likelihood estimation for
  Bayesian phylogenetic model selection.* Systematic Biology 60(2):150–160,
  2011. [doi:10.1093/sysbio/syq085](https://doi.org/10.1093/sysbio/syq085)

### Optimization and variational inference

- **L-BFGS** (`optimize/map-estimate`, `laplace`, `pathfinder`) — D. C. Liu,
  J. Nocedal. *On the limited memory BFGS method for large scale
  optimization.* Mathematical Programming 45:503–528, 1989.
  [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116); J. Nocedal,
  S. J. Wright. *Numerical Optimization.* 2nd ed., Springer, 2006.
- **Pathfinder** (`optimize/pathfinder`) — L. Zhang, B. Carpenter,
  A. Gelman, A. Vehtari. *Pathfinder: Parallel quasi-Newton variational
  inference.* JMLR 23(306):1–49, 2022.
  [arXiv:2108.03782](https://arxiv.org/abs/2108.03782)
- **Pareto-smoothed importance sampling** (Pathfinder's resampling,
  `diagnostics/psis`, LOO) — A. Vehtari, D. Simpson, A. Gelman, Y. Yao,
  J. Gabry. *Pareto Smoothed Importance Sampling.* JMLR 25(72), 2024.
  [arXiv:1507.02646](https://arxiv.org/abs/1507.02646)
- **BBVI** (`bbvi-infer`) — R. Ranganath, S. Gerrish, D. Blei. *Black Box
  Variational Inference.* AISTATS 2014.
  [arXiv:1401.0118](https://arxiv.org/abs/1401.0118)
- **Adam** (`learn/maximize-evidence`) — D. P. Kingma, J. Ba. *Adam: A
  Method for Stochastic Optimization.* ICLR 2015.
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

## Steering and twisted SMC

- **Twisted particle filters** — N. Whiteley, A. Lee. *Twisted particle
  filters.* Annals of Statistics 42(1):115–141, 2014.
  [arXiv:1210.0220](https://arxiv.org/abs/1210.0220)
- **Twisted SMC for language models** — S. Zhao, R. Brekelmans, A. Makhzani,
  R. Grosse. *Probabilistic Inference in Language Models via Twisted
  Sequential Monte Carlo.* ICML 2024.
  [arXiv:2404.17546](https://arxiv.org/abs/2404.17546)
- **SMC steering** — A. K. Lew, T. Zhi-Xuan, G. Grand, V. K. Mansinghka.
  *Sequential Monte Carlo Steering of Large Language Models using
  Probabilistic Programs.* 2023.
  [arXiv:2306.03081](https://arxiv.org/abs/2306.03081)
- **Controlled generation by SMC (genlm)** — J. Loula, B. LeBrun, L. Du,
  B. Lipkin, C. Pasti, G. Grand, T. Liu, Y. Emara, M. Freedman, J. Eisner,
  R. Cotterell, V. Mansinghka, A. K. Lew, T. Vieira, T. J. O'Donnell.
  *Syntactic and Semantic Control of Large Language Models via Sequential
  Monte Carlo.* ICLR 2025. [arXiv:2504.13139](https://arxiv.org/abs/2504.13139) ·
  [genlm-control](https://github.com/genlm/genlm-control)
- **The reward as a tilt** — T. Korbak, E. Perez, C. L. Buckley. *RL with
  KL penalties is better viewed as Bayesian inference.* Findings of EMNLP
  2022.

## Causality

- J. Pearl. *Causality: Models, Reasoning, and Inference.* 2nd ed., 2009 —
  interventions, the three-step counterfactual (abduction, action,
  prediction), the probability of necessity.

## Checking models and samplers

- **Bayesian workflow** — A. Gelman, A. Vehtari, D. Simpson, C. C. Margossian,
  B. Carpenter, Y. Yao, L. Kennedy, J. Gabry, P.-C. Bürkner, M. Modrák.
  *Bayesian Workflow.* 2020. [arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
  — the outline of the [workflow guide](workflow.md).
- **R-hat, ESS, MCSE** — A. Vehtari, A. Gelman, D. Simpson, B. Carpenter,
  P.-C. Bürkner. *Rank-normalization, folding, and localization: an improved
  R̂ for assessing convergence of MCMC.* Bayesian Analysis 16(2):667–718,
  2021. [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221) — the
  diagnostics `foerster.diagnostics` computes.
- **PSIS-LOO and WAIC** — A. Vehtari, A. Gelman, J. Gabry. *Practical
  Bayesian model evaluation using leave-one-out cross-validation and WAIC.*
  Statistics and Computing 27:1413–1432, 2017.
  [doi:10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4) — what `pointwise-log-likelihood` is the
  input of.
- **posteriordb** (the reference posteriors of `clojure -M:test:posteriordb`)
  — M. Magnusson, J. Torgander, P.-C. Bürkner, L. Zhang, B. Carpenter,
  A. Vehtari. *posteriordb: Testing, Benchmarking and Developing Bayesian
  Inference Algorithms.* AISTATS 2025, PMLR 258:1198–1206.
  [proceedings.mlr.press/v258/magnusson25a](https://proceedings.mlr.press/v258/magnusson25a.html)
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
