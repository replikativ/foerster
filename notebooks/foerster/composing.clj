;; # Composing Inference

;; Foerster's inference is built from a few pieces that plug into each other
;; (see [the programming model](https://github.com/replikativ/foerster/blob/main/doc/programming-model.md)):
;;
;; - a **proposal** says where a latent site's value comes from;
;; - a **kernel** is a Markov move that leaves the posterior invariant;
;; - a **pair** is a move with auxiliary randomness (SMCP3, transport, involutions);
;; - a **population** is weighted particles, which an **inference program** advances,
;;   resamples and moves;
;; - a **result** says what it vouches for (its contract).
;;
;; This notebook runs each piece on a model whose posterior is known exactly,
;; and plugs the same pieces into several places.

(ns foerster.composing
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.contract :as contract]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.moves.pair :as pair]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.proposal :as proposal]
            [org.replikativ.foerster.runtime.trace :as trace]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def world (sp/create-execution-context))

;; `run` evaluates an inference — its model included — in that world and
;; waits for the result.
(defmacro run [inference] `(sp/with-context world @~inference))

;; ## A model with a known posterior
;;
;; Two independent means, a ~ N(0, 1) and b ~ N(0, 1), each observed eight
;; times with N(·, 1) noise: each posterior is N(Σy/9, 1/9).

(def ys-a [1.4 0.6 1.1 0.9 1.7 1.2 0.8 1.3])
(def ys-b [-0.4 -1.1 -0.7 -0.2 -0.9 -0.6 -1.3 -0.5])

(defn model []
  (spin (let [a (sample (dist/normal 0.0 1.0) :id :a)
              b (sample (dist/normal 0.0 1.0) :id :b)]
          (doseq [[i y] (map-indexed vector ys-a)] (observe (dist/normal a 1.0) y :id [:ya i]))
          (doseq [[i y] (map-indexed vector ys-b)] (observe (dist/normal b 1.0) y :id [:yb i]))
          [a b])))

(def exact {:a (/ (reduce + ys-a) 9.0) :b (/ (reduce + ys-b) 9.0) :sd (/ 1.0 3.0)})

(defn means
  "The weighted posterior means of a and b."
  [measure]
  (let [ps (m/get-particles measure)
        ws (weights/normalize-log-weights (mapv second ps))
        vs (mapv (comp m/get-value first) ps)]
    {:a (reduce + (map #(* %1 (first %2)) ws vs))
     :b (reduce + (map #(* %1 (second %2)) ws vs))}))

exact

;; ## Proposals: where a site's value comes from
;;
;; Importance sampling from the prior wastes most draws: the data pull the
;; posterior away from it. A `:proposal` draws `:a` and `:b` near the data
;; instead; the weights correct for it (log p − log q), so the target is the
;; same.

(def near-data
  (proposal/only #{:a :b} (proposal/from-fn
                           (fn [{:keys [address]}]
                             (let [q (dist/normal (get exact address) 0.5) v (dist/draw q)]
                               {:value v :log-q (dist/logpdf q v)}))
                           (fn [{:keys [address]} v] (dist/logpdf (dist/normal (get exact address) 0.5) v)))))

(let [prior (run (infer/importance-sampling (model) 2000 {:seed 1}))
      guided (run (infer/importance-sampling (model) 2000 {:seed 1 :proposal near-data}))]
  {:prior {:ess (m/effective-sample-size prior) :means (means prior)}
   :guided {:ess (m/effective-sample-size guided) :means (means guided)}})

;; The guided run keeps many more effective particles. A proposal of your own
;; (`from-fn`) is your obligation, and the result says so; the built-in
;; proposals (`prior`, `distribution`, `random-walk`) are right by
;; construction:

{:from-fn (contract/normalizer-status (run (infer/importance-sampling (model) 100 {:seed 2 :proposal near-data})))
 :distribution (contract/normalizer-status
                (run (infer/importance-sampling (model) 100 {:seed 2 :proposal (proposal/distribution (dist/normal 0.0 2.0))})))}

;; ## Kernels compose
;;
;; A kernel moves some sites. `:only` says which; `cycle` runs kernels in
;; turn. A chain refuses a kernel that leaves a site unmoved, naming it:

(try (run (infer/kernel-infer (model) (k/single-site-mh-kernel 100 {:only #{:a}}) 2 {:seed 3}))
     (catch Exception e (ex-message (or (ex-cause e) e))))

(def per-site
  (k/cycle 2000 [(k/single-site-mh-kernel 1 {:only #{:a}})
                 (k/single-site-mh-kernel 1 {:only #{:b}})]
           {:samples :all :burn 200}))

(means (run (infer/kernel-infer (model) per-site 2 {:seed 4})))

;; ## A kernel of your own
;;
;; `k/kernel` makes a kernel from a step over traces — here MH with a random
;; walk, passing the context (target, replay frontier, ownership) on to
;; `mh-step`. It composes with the built-in kernels.

(def walk
  (k/kernel 1 (fn [t context] (trace/mh-step t (assoc context :propose (proposal/random-walk 0.4))))
            {:retains? true :tempering? true :id :walk}))

(means (run (infer/kernel-infer (model) (k/cycle 2000 [walk (k/single-site-mh-kernel 1)] {:samples :all :burn 200})
                                2 {:seed 5})))

;; ## One kernel, many places
;;
;; The same kernel rejuvenates SMC after each resampling and moves tempered
;; SMC's particles between temperatures:

(def mix (k/cycle 1 [(k/single-site-mh-kernel 1) walk]))

{:smc (means (run (infer/smc-infer (model) 500 {:seed 6 :anchors :all :rejuvenate {:moves 2 :kernel mix}})))
 :tempered (means (run (infer/tempered-infer (model) 500 {:seed 7 :kernel mix})))}

;; ## Pairs: moves with auxiliary randomness
;;
;; An involutive move — draw u, map (a, u) to (a + u, −u) — is an MCMC move
;; and, as a pair, an SMCP3 move-reweight step:

(def shift
  (let [q (dist/normal 0.0 0.3)]
    {:propose (fn [_] (let [u (dist/draw q)] {:aux u :log-q (dist/logpdf q u)}))
     :log-q (fn [_ u] (dist/logpdf q u))
     :involution (fn [{a :a} u] {:choices {:a (+ a u)} :aux (- u) :log-jacobian 0.0})}))

{:mcmc (means (run (infer/kernel-infer (model)
                                       (k/cycle 2000 [(k/involutive-kernel 1 shift) (k/single-site-mh-kernel 1 {:only #{:b}})]
                                                {:samples :all :burn 200})
                                       2 {:seed 8})))
 :smcp3 (means (run (infer/smc-infer (model) 500 {:seed 9 :anchors #{:a} :smcp3 (pair/from-involution shift)})))}

;; ## An inference program
;;
;; When no built-in driver fits, a program composes the population
;; operations itself. This one anneals from the prior to the posterior,
;; resampling when the effective sample size halves, and moves every
;; particle with the per-site kernel at each temperature:

(defn anneal [model]
  (ip/run {:seed 10}
          (fn [scope]
            (spin
             (loop [p (await (ip/initialize scope (ip/trace-target model) 400))
                    betas [0.1 0.3 0.6 1.0]]
               (if-not (seq betas)
                 (ip/finish scope p)
                 (let [p (ip/advance scope p (ip/temperature (:target p) (first betas)))
                       p (if (< (ip/ess p) 200) (await (ip/resample scope p)) p)
                       p (await (ip/move scope p (k/cycle 3 [(k/single-site-mh-kernel 1 {:only #{:a}})
                                                             (k/single-site-mh-kernel 1 {:only #{:b}})])))]
                   (recur p (next betas)))))))))

(let [r (run (anneal (model)))]
  {:means (means r) :contract (:inference-contract r)})

;; Each piece is checked where it is taken: a proposal that does not return
;; a density, a kernel that cannot target a temperature, a pair without its
;; backward kernel are refused when inference starts, not after it ran.
