(ns org.replikativ.foerster.inference.bbvi
  "Black-box variational inference (Ranganath, Gerrish & Blei 2014): a
  mean-field q learned by score-function ascent on the ELBO, each iteration
  importance sampling from q on savepoint SMC."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as-alias core]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.inference.gradient :as grad]
            [org.replikativ.foerster.inference.worlds :refer [check-options! particle-options particles]]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.foerster.runtime.canonical :refer [with-frozen-caller]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]
            [clojure.set :as set]))

(defn- variational-draw
  "The proposal of a savepoint BBVI iteration: a latent site whose q (the
  site's prior, the first time its address is seen) has a gradient draws from
  q; `foerster.runtime.trace/policy` then weights it by p/q."
  [q-dists]
  (fn [sp _old-entry]
    (let [address (:savepoint/address sp)
          prior (:dist (:savepoint/payload sp))
          q (get (swap! q-dists #(if (contains? % address) % (assoc % address prior))) address)]
      (when (grad/has-gradient? q)
        (let [v (dist/draw q)]
          {:value v :log-proposal (dist/logpdf q v)})))))

(defn- q-gradients
  "{address ∇log q(value)} of the latent sites of a Sample's trace that
  `qs` has a differentiable q for."
  [qs trace]
  (into {} (keep (fn [[address {:keys [value observed?]}]]
                   (let [q (get qs address)]
                     (when (and (not observed?) q (grad/has-gradient? q))
                       [address (grad/compute-gradient q value)]))))
        trace))

(defn- optimal-scaling
  "Control variate coefficient Cov(f,g)/Var(g)."
  [f g]
  (let [n (count f)
        f-bar (/ (reduce + f) n)
        g-bar (/ (reduce + g) n)
        g-centered (mapv #(- % g-bar) g)]
    (/ (reduce + 1e-12 (map * (map #(- % f-bar) f) g-centered))
       (reduce + 1e-12 (map * g-centered g-centered)))))

(defn- aggregate-gradients
  "Per address, the score-function ELBO gradient
     (1/n) Σ (log w_i − a*) ∇log q(z_i),   log w = log p(x, y) − log q(x)
   with the variance-minimizing control variate a* = Cov(f, g)/Var(g),
   f = log w · g (Ranganath et al. 2014)."
  [particle-gradients particle-log-weights]
  (let [all-addrs (reduce set/union (map #(set (keys %)) particle-gradients))]
    (reduce
     (fn [result addr]
       (let [valid (filter (fn [[grads lw]] (and (contains? grads addr) (grad/finite? lw)))
                           (map vector particle-gradients particle-log-weights))
             n (count valid)]
         (if (< n 2)
           result
           (let [grads (mapv #(get (first %) addr) valid)
                 lws (mapv second valid)]
             (assoc result addr
                    (mapv (fn [dim]
                            (let [g (mapv #(nth % dim) grads)
                                  f (mapv * lws g)
                                  a (optimal-scaling f g)]
                              (/ (reduce + (map (fn [x y] (- x (* a y))) f g)) n)))
                          (range (count (first grads)))))))))
     {}
     all-addrs)))

(defn- update-variational-dists!
  "One ascent step on every q with a gradient. `adagrad`: true accumulates
   squared gradients (AdaGrad), a number γ decays them (RMSprop), false
   takes plain steps of size `lr`."
  [q-dists accumulators gradients lr adagrad]
  (doseq [[addr g] gradients]
    (let [dist (get @q-dists addr)]
      (when (grad/has-gradient? dist)
        (let [g2 (mapv #(* % %) g)
              prev (get @accumulators addr)
              acc (cond (nil? prev) g2
                        (number? adagrad) (mapv #(+ (* adagrad %1) (* (- 1.0 adagrad) %2)) prev g2)
                        :else (mapv + prev g2))
              rate (if adagrad (mapv #(/ lr (+ 1e-8 (Math/sqrt %))) acc) lr)]
          (swap! accumulators assoc addr acc)
          (swap! q-dists assoc addr (grad/grad-step dist g rate)))))))

(declare bbvi)

(defn bbvi-infer
  "Black Box Variational Inference (Ranganath et al., AISTATS 2014).

   Learns a mean-field q(z) = Π_addr q_addr by stochastic ascent on the ELBO
   with the score-function estimator and control variates: `num-iterations`
   updates, each from `num-particles` programs that sample latents from q
   and weight by p(x, y)/q(x). Each site's q starts as the prior it has when
   first reached and from then on moves by gradient only: mean-field, it
   does not follow a prior that depends on other latents.

   Args:
     model-task, num-particles, num-iterations
     opts: :base-lr (1.0) — step size at iteration t is
           base-lr / (t+1)^robbins-monro
           :robbins-monro (0.0)
           :adagrad (true) — true: AdaGrad, γ ∈ (0,1): RMSprop with decay γ,
           false: plain steps
           :executor

   Returns: Spin<EmpiricalMeasure> — `num-particles` importance-weighted
   samples from the final q (with 0 iterations: from the priors); the learned
   q is under `:variational-dists` (see `get-variational-dists`)."
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (-> particle-options
                           (disj :policy :resample-threshold)
                           (into #{:base-lr :robbins-monro :adagrad})))
  (with-frozen-caller opts #(bbvi model-task num-particles num-iterations %)))

(defn- bbvi
  [model-task num-particles num-iterations opts]
  (spin
   (let [base-lr (or (:base-lr opts) 1.0)
         robbins-monro (or (:robbins-monro opts) 0.0)
         adagrad (get opts :adagrad true)
         q-dists (atom {})
         accumulators (atom {})]
     (loop [iteration 0]
       (let [qs @q-dists
             measure (await (particles model-task num-particles
                                       (assoc opts
                                              :resample-threshold 0.0
                                              :policy (itrace/policy {:draw (variational-draw q-dists)}))))]
         (if (>= iteration num-iterations)
           (assoc measure :variational-dists @q-dists)
           (let [ps (m/get-particles measure)
                 grads (mapv (fn [[c _]] (q-gradients (merge @q-dists qs) (m/get-trace c)))
                             ps)]
             (update-variational-dists! q-dists accumulators
                                        (aggregate-gradients grads (mapv second ps))
                                        (/ base-lr (Math/pow (inc iteration) robbins-monro))
                                        adagrad)
             (recur (inc iteration)))))))))

(defn get-variational-dists
  "The learned {address -> distribution} of a `bbvi-infer` result."
  [measure]
  (:variational-dists measure))
