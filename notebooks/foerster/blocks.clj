;; # Blocks and HMC

;; Single-site Metropolis–Hastings changes one latent variable at a time.
;; When many continuous latents are correlated — the weights of a regression,
;; the parameters of a dynamical system — one-at-a-time moves crawl.
;; Hamiltonian Monte Carlo moves all of them together, along the gradient of
;; the log density. In foerster, the latents HMC moves together form a
;; **block**: one site whose value is a vector.

(ns foerster.blocks
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

(def world (sp/create-execution-context))

;; ## Data
;;
;; Forty binary outcomes y from one feature x, drawn from a logistic model
;; with intercept 0.5 and slope −1.2.

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(random/set-seed! 9)

(def xs (vec (repeatedly 40 #(dist/draw (dist/normal 0.0 1.0)))))
(def ys (mapv #(dist/draw (dist/bernoulli (sigmoid (+ 0.5 (* -1.2 %))))) xs))

;; ## A block
;;
;; A block is a **description** — data: its latents, their shapes and
;; support, and what its density covers — and **capabilities** — functions
;; over a flat `double[]` θ and the site's inputs: `:log-density` and
;; `:value+grad` (the value and its gradient). Here θ = [b0 b1], with a
;; N(0, 2²) prior on each and the Bernoulli likelihood of the data:

(defn logistic-log-density+gradient [^doubles b {:keys [xs ys]}]
  (let [b0 (aget b 0) b1 (aget b 1)]
    (loop [i 0
           lp (- (/ (+ (* b0 b0) (* b1 b1)) 8.0))
           g0 (- (/ b0 4.0))
           g1 (- (/ b1 4.0))]
      (if (= i (count xs))
        [lp (double-array [g0 g1])]
        (let [x (nth xs i) y (nth ys i)
              eta (+ b0 (* b1 x))
              r (- y (sigmoid eta))]
          (recur (inc i)
                 (+ lp (- (* y eta) (Math/log1p (Math/exp eta))))
                 (+ g0 r)
                 (+ g1 (* r x))))))))

(def logistic
  (block/block {:block/id :logistic
                :block/latents [{:name :beta :shape [2] :support :real}]
                :block/target :complete-conditional}
               {:log-density (fn [b inputs] (first (logistic-log-density+gradient b inputs)))
                :value+grad logistic-log-density+gradient}))

;; `:complete-conditional` says the density covers every factor the latents
;; touch: their priors and every observation that depends on them. Those
;; observations are inside the block, so the model does not observe them
;; again. In the model the block is one site:

(defn regression []
  (spin (sample (block/block-dist logistic {:xs xs :ys ys}) :id :beta :init [0.0 0.0])))

;; A block that cannot be drawn from (no `:sample` capability) starts from
;; its `:init`.

;; ## HMC
;;
;; `hmc-kernel` runs HMC-within-Gibbs: it moves block sites by HMC and any
;; other latent site by Metropolis–Hastings. Two chains, 1500 moves each,
;; leapfrog steps of 0.1:

(random/set-seed! 1)

(def posterior
  (sp/with-context world
    @(infer/kernel-infer (regression)
                         (k/hmc-kernel 1500 {:step-size 0.1 :steps 8 :samples :all :burn 200})
                         2)))

(def draws (mapv (comp m/get-value first) (m/get-particles posterior)))

(-> (tc/dataset {:b0 (map first draws) :b1 (map second draws)})
    (plotly/layer-point {:=x :b0 :=y :b1 :=mark-opacity 0.3}))

;; ## Checking it
;;
;; With two parameters the posterior can also be computed on a grid, which
;; is exact up to the grid's resolution:

(defn grid-mean []
  (let [n 161 h (/ 8.0 (dec n))
        points (for [i (range n) j (range n)] [(+ -4.0 (* i h)) (+ -4.0 (* j h))])
        lps (mapv #((block/capability logistic :log-density) (double-array %) {:xs xs :ys ys}) points)
        top (apply max lps)
        ws (mapv #(Math/exp (- % top)) lps)
        z (reduce + ws)]
    [(/ (reduce + (map #(* %1 (first %2)) ws points)) z)
     (/ (reduce + (map #(* %1 (second %2)) ws points)) z)]))

{:hmc [(/ (reduce + (map first draws)) (count draws))
       (/ (reduce + (map second draws)) (count draws))]
 :grid (grid-mean)}

;; A block's gradient is easy to get wrong by hand; compare it with finite
;; differences before trusting a chain:

(let [lp (block/capability logistic :log-density)
      [_ g] ((block/capability logistic :value+grad) (double-array [0.3 -0.7]) {:xs xs :ys ys})
      h 1e-6
      fd (fn [j] (/ (- (lp (double-array (update [0.3 -0.7] j + h)) {:xs xs :ys ys})
                       (lp (double-array (update [0.3 -0.7] j - h)) {:xs xs :ys ys}))
                    (* 2 h)))]
  {:analytic (vec g) :finite-differences [(fd 0) (fd 1)]})

;; ## When the density is incomplete
;;
;; HMC's proposal follows the block's density, but foerster accepts or
;; rejects it on the **full** log joint of the program's trace. A block whose
;; density leaves out a factor — observed elsewhere in the model — still
;; samples the right posterior; it just mixes worse, and each step records
;; it. So a block's density is a statement about efficiency, not about
;; correctness.

;; ## Compiled blocks
;;
;; The capabilities are plain functions over arrays. They can come from
;; anywhere; [foerster-raster](https://github.com/replikativ/foerster-raster)
;; compiles them from [raster](https://github.com/replikativ/raster) `deftm`
;; log densities, with gradients from raster's reverse-mode automatic
;; differentiation, to JVM bytecode and (by raster's backends) WASM and GPU
;; kernels.
