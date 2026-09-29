;; # Programmable Inference

;; The inference algorithms of the other notebooks are ready-made. When a
;; model needs a custom move — a proposal that knows its structure, a jump
;; between two explanations — foerster offers the building blocks they are
;; made of. The interface follows [Gen](https://www.gen.dev/)'s *generative
;; function interface*: operations on **traces**, the record of one run of
;; a program with every random choice at its address.

(ns foerster.programmable
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.involutive :as inv]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as trace]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.select :as select]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def world (sp/create-execution-context))
(random/set-seed! 7)

;; The operations are CPS operations; at the REPL we wait for them:

(defn await-cps [operation]
  (let [p (promise)]
    (operation #(deliver p [:ok %]) #(deliver p [:error %]))
    (let [[status v] (deref p 60000 [:error (ex-info "timed out" {})])]
      (if (= :ok status) v (throw v)))))

;; ## A model with addresses
;;
;; x ~ N(0, 1), z | x ~ N(x, 1), and y = 2 observed from N(z, 1). Sites are
;; named with `:id`; the posterior of x is N(2/3, 2/3).

(defn model []
  (sp/with-context world
    (spin
     (let [x (sample (dist/normal 0.0 1.0) :id :x)
           z (sample (dist/normal x 1.0) :id :z)]
       (observe (dist/normal z 1.0) 2.0 :id :y)
       x))))

;; ## The operations
;;
;; **simulate** runs the program, every choice from its prior:

(def t (await-cps (gfi/simulate (model))))

(trace/choices t)

;; A trace holds its world: close it when done (`gfi/close!`). The ones
;; below are closed at the end of the notebook.
;;
;; **generate** runs it with some choices constrained, and returns the trace
;; with an importance **weight**: the log density of the constrained and
;; observed sites. Averaging `exp(weight)` over runs with no constraints
;; estimates the evidence.

(def g (await-cps (gfi/generate (model) {:z 1.5})))

{:choices (trace/choices (:trace g)) :weight (:weight g)}

;; **assess** scores a complete assignment of the latent sites — the log
;; joint density, observations included:

(:weight (await-cps (gfi/assess (model) {:x 0.5 :z 1.5})))

;; **update** changes some choices of an existing trace, keeps the others
;; (rescored under their distributions as they are now), and returns the
;; weight of the change and the values it overwrote:

(let [{t' :trace w :weight d :discard} (await-cps (gfi/update (:trace g) {:x -0.5}))]
  {:choices (trace/choices t') :weight w :discard d})

;; **regenerate** proposes fresh values for a *selection* of sites from
;; their priors, keeping the rest. Selections are sets of addresses, or
;; spindel **selectors** — by id, path, prefix or site kind, combined with
;; `union`, `intersection` and `complement*`:

(let [{t' :trace} (await-cps (gfi/regenerate t (select/id :z)))]
  {:before (trace/choices t) :after (trace/choices t')})

;; ## MH by selection
;;
;; `gfi/mh` is a Metropolis–Hastings move that regenerates a selection and
;; accepts or rejects. Alternating between x and z gives a Gibbs-like
;; sampler for the posterior of x:

(def xs
  (loop [t (await-cps (gfi/simulate (model))), i 0, xs []]
    (if (= i 4000)
      (do (await-cps (gfi/close! t)) xs)
      (let [{t' :trace} (await-cps (gfi/mh t (if (even? i) #{:x} #{:z})))]
        (recur t' (inc i) (if (> i 1000) (conj xs (:trace/result t')) xs))))))

(let [n (count xs) mu (/ (reduce + xs) n)]
  {:mean mu
   :variance (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n)
   :exact-mean 2/3 :exact-variance 2/3})

;; ## Involutive MCMC
;;
;; Some moves are not "redraw these sites": a jump that scales a value, swaps
;; two components, or splits one into two. Involutive MCMC (Cusumano-Towner,
;; Lew and Mansinghka, 2020) describes such a move by an auxiliary draw and
;; an **involution** — a map that is its own inverse — on (choices, aux),
;; with the log absolute Jacobian determinant of the map.
;;
;; A rate λ ~ Gamma(2, 1) with a count of 5 observed from Poisson(λ): the
;; posterior is Gamma(7, rate 2), mean 3.5. A scale move multiplies λ by s
;; and replaces s by 1/s:

(defn gamma-poisson []
  (sp/with-context world
    (spin (let [l (sample (dist/gamma 2.0 1.0) :id :l)]
            (observe (dist/poisson l) 5 :id :y)
            l))))

(defn scale-move [with-jacobian?]
  (let [log-q (fn [_ s] (- (Math/log (* 2.0 s))))]      ; log s ~ U(−1, 1)
    {:propose (fn [_] (let [s (Math/exp (dist/draw (dist/uniform -1.0 1.0)))]
                        {:aux s :log-q (log-q nil s)}))
     :log-q log-q
     :involution (fn [{l :l} s]
                   {:choices {:l (* l s)}
                    :aux (/ 1.0 s)
                    ;; |det ∂(λs, 1/s)/∂(λ, s)| = 1/s
                    :log-jacobian (if with-jacobian? (- (Math/log s)) 0.0)})}))

(defn chain-mean [move]
  (loop [t (await-cps (gfi/simulate (gamma-poisson))), i 0, ls []]
    (if (= i 4000)
      (do (await-cps (gfi/close! t)) (/ (reduce + ls) (count ls)))
      (let [{t' :trace} (await-cps (inv/step t move))]
        (recur t' (inc i) (if (>= i 500) (conj ls (get-in t' [:trace/entries :l :value])) ls))))))

{:with-jacobian (chain-mean (scale-move true))
 :without (chain-mean (scale-move false))
 :exact 3.5}

;; Leaving the Jacobian out targets a different distribution — the chain
;; settles somewhere else. `inv/fd-log-jacobian` computes it numerically
;; for a check.

;; ## Cleaning up

(await-cps (gfi/close! t))
(await-cps (gfi/close! (:trace g)))
