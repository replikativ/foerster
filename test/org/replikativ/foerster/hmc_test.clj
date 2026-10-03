(ns org.replikativ.foerster.hmc-test
  "HMC on block sites against analytic and quadrature posteriors, with
  pure-Clojure reference blocks (the oracles of the spindel ↔ raster block
  contract)."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :refer [run-infer weighted-values w-mean-sd]]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.hmc :as hmc]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.foerster.dist :as dist]))

;; --- reference blocks --------------------------------------------------------

(defn- gaussian-block
  "μ ∈ R^d, μ_j ~ N(0, s0²), y_ij ~ N(μ_j, s²). With `observations?` false
  only the prior: an incomplete target when the y are observed elsewhere."
  [d observations?]
  (let [lp+grad (fn [^doubles mu {:keys [ys s0 s]}]
                  (let [ys (if observations? ys [])
                        lp (- (+ (reduce + (for [j (range d)]
                                             (/ (* (aget mu j) (aget mu j)) (* 2 s0 s0))))
                                 (reduce + 0.0 (for [y ys j (range d)]
                                                 (let [r (- (nth y j) (aget mu j))]
                                                   (/ (* r r) (* 2 s s)))))))
                        g (double-array
                           (for [j (range d)]
                             (+ (- (/ (aget mu j) (* s0 s0)))
                                (reduce + 0.0 (for [y ys]
                                                (/ (- (nth y j) (aget mu j)) (* s s)))))))]
                    [lp g]))]
    (block/block {:block/id :gaussian
                  :block/latents [{:name :mu :shape [d] :support :real}]
                  :block/target (if observations? :complete-conditional :prior)}
                 {:log-density (fn [mu inputs] (first (lp+grad mu inputs)))
                  :value+grad lp+grad})))

(def ^:private gauss-inputs
  (let [rng (java.util.Random. 5)]
    {:s0 3.0 :s 1.0
     :ys (vec (repeatedly 20 (fn [] [(+ 1.0 (.nextGaussian rng)) (+ -2.0 (.nextGaussian rng))])))}))

(defn- gauss-posterior
  "[[mean sd] per dimension] of the Gaussian block's posterior."
  [{:keys [ys s0 s]} d]
  (let [prec (+ (/ 1.0 (* s0 s0)) (/ (count ys) (* s s)))]
    (vec (for [j (range d)]
           [(/ (/ (reduce + (map #(nth % j) ys)) (* s s)) prec) (/ 1.0 (Math/sqrt prec))]))))

(defn- sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(def ^:private logistic-block
  ;; β = [b0 b1], β ~ N(0, 2²), y_i ~ Bernoulli(σ(b0 + b1 x_i))
  (let [lp+grad (fn [^doubles b {:keys [xs ys]}]
                  (let [b0 (aget b 0) b1 (aget b 1)]
                    (loop [i 0 lp (- (/ (+ (* b0 b0) (* b1 b1)) 8.0))
                           g0 (- (/ b0 4.0)) g1 (- (/ b1 4.0))]
                      (if (= i (count xs))
                        [lp (double-array [g0 g1])]
                        (let [x (nth xs i) y (nth ys i) eta (+ b0 (* b1 x))
                              r (- y (sigmoid eta))]
                          (recur (inc i)
                                 (+ lp (- (* y eta) (Math/log1p (Math/exp eta))))
                                 (+ g0 r) (+ g1 (* r x))))))))]
    (block/block {:block/id :logistic
                  :block/latents [{:name :beta :shape [2] :support :real}]
                  :block/target :complete-conditional}
                 {:log-density (fn [b inputs] (first (lp+grad b inputs)))
                  :value+grad lp+grad})))

(def ^:private logistic-inputs
  (let [rng (java.util.Random. 9)
        xs (vec (repeatedly 40 #(.nextGaussian rng)))]
    {:xs xs :ys (mapv #(if (< (.nextDouble rng) (sigmoid (+ 0.5 (* -1.2 %)))) 1 0) xs)}))

(defn- quadrature-mean
  "Posterior mean of β by a grid over [-4, 4]²."
  [inputs]
  (let [n 241 h (/ 8.0 (dec n))
        pts (for [i (range n) j (range n)] [(+ -4.0 (* i h)) (+ -4.0 (* j h))])
        lps (mapv #((block/capability logistic-block :log-density) (double-array %) inputs) pts)
        mx (apply max lps)
        ws (mapv #(Math/exp (- % mx)) lps)
        z (reduce + ws)]
    [(/ (reduce + (map #(* %1 (first %2)) ws pts)) z)
     (/ (reduce + (map #(* %1 (second %2)) ws pts)) z)]))

;; --- helpers -----------------------------------------------------------------

(defn- block-model [b inputs init]
  (fn [] (spin (sample (block/block-dist b inputs) :id :theta :init init))))

(defn- chain-moments
  "[[mean sd] per dimension] of an HMC chain's samples."
  [b inputs init {:keys [iterations step-size steps seed]}]
  (let [wv (weighted-values
            (run-infer seed #(infer/kernel-infer ((block-model b inputs init))
                                                 (k/hmc-kernel iterations {:step-size step-size :steps steps
                                                                           :samples :all :burn 100})
                                                 2 {})))
        d (count init)]
    (vec (for [j (range d)]
           (let [xs (map #(nth (first %) j) wv)
                 ws (map second wv)
                 mean (reduce + (map * xs ws))
                 var (reduce + (map #(* %2 (let [r (- %1 mean)] (* r r))) xs ws))]
             [mean (Math/sqrt var)])))))

(defn- fd-check
  "Max |analytic − central-difference| over the gradient at `n` points."
  [b inputs d n seed]
  (let [rng (java.util.Random. seed)
        lp (block/capability b :log-density)
        vg (block/capability b :value+grad)]
    (reduce max
            (for [_ (range n)
                  :let [x (vec (repeatedly d #(.nextGaussian rng)))
                        [_ g] (vg (double-array x) inputs)]
                  j (range d)
                  :let [h 1e-6
                        up (lp (double-array (update x j + h)) inputs)
                        down (lp (double-array (update x j - h)) inputs)]]
              (Math/abs (- (aget ^doubles g j) (/ (- up down) (* 2 h))))))))

;; --- tests -------------------------------------------------------------------

(deftest reference-gradients-match-finite-differences
  (is (< (fd-check (gaussian-block 2 true) gauss-inputs 2 20 1) 1e-4))
  (is (< (fd-check logistic-block logistic-inputs 2 20 2) 1e-4)))

(deftest hmc-recovers-a-gaussian-posterior
  (let [truth (gauss-posterior gauss-inputs 2)
        got (chain-moments (gaussian-block 2 true) gauss-inputs [0.0 0.0]
                           {:iterations 1500 :step-size 0.1 :steps 8 :seed 3})]
    (doseq [[[m s] [m' s']] (map vector truth got)]
      (is (< (Math/abs (- m m')) 0.03) (str "mean " m' " vs " m))
      (is (< (Math/abs (- s s')) 0.03) (str "sd " s' " vs " s)))))

(deftest hmc-recovers-a-logistic-posterior
  (let [truth (quadrature-mean logistic-inputs)
        got (mapv first (chain-moments logistic-block logistic-inputs [0.0 0.0]
                                       {:iterations 1500 :step-size 0.15 :steps 10 :seed 4}))]
    (doseq [[m m'] (map vector truth got)]
      (is (< (Math/abs (- m m')) 0.06) (str "mean " m' " vs quadrature " m)))))

(deftest an-incomplete-target-is-exact-and-reported
  ;; The block holds only the prior; the observations are sites of their own.
  ;; Acceptance on the full joint keeps the chain exact; the steps say the
  ;; target was incomplete.
  (let [b (gaussian-block 2 false)
        {:keys [ys] :as inputs} gauss-inputs
        root (ctx/create-execution-context)
        session (sp/open! root {:seed 11 :fork-opts {:systems :none}})
        flags (atom #{})
        samples (atom [])
        await-cps (fn [op] (let [p (promise)]
                             (op #(deliver p [:ok %]) #(deliver p [:err %]))
                             (let [[k v] (deref p 60000 [:err (ex-info "timeout" {})])]
                               (if (= k :ok) v (throw v)))))]
    (try
      (let [model (binding [ec/*execution-context* root]
                    (spin (let [mu (sample (block/block-dist b inputs) :id :mu :init [0.0 0.0])]
                            (loop [i 0]
                              (when (< i (count ys))
                                (let [y (nth ys i)]
                                  (observe (dist/normal (nth mu 0) 1.0) (nth y 0) :id [:y i 0])
                                  (observe (dist/normal (nth mu 1) 1.0) (nth y 1) :id [:y i 1]))
                                (recur (inc i))))
                            mu)))
            t0 (await-cps (trace/run session model (itrace/policy {:init? true}) {:anchor? itrace/anchor?}))]
        (await-cps (itrace/mh-chain t0 1500
                                    {:step (hmc/within-gibbs {:step-size 0.1 :steps 8})
                                     :on-step (fn [{:keys [trace incomplete-target?]}]
                                                (swap! flags conj incomplete-target?)
                                                (swap! samples conj (:trace/result trace)))}))
        (is (contains? @flags true))
        (let [kept (drop 100 @samples)]
          (doseq [[j [m s]] (map-indexed vector (gauss-posterior inputs 2))]
            (let [xs (map #(nth % j) kept)
                  mean (/ (reduce + xs) (count xs))
                  sd (Math/sqrt (/ (reduce + (map #(let [r (- % mean)] (* r r)) xs)) (count xs)))]
              (is (< (Math/abs (- mean m)) 0.05) (str "mean " mean " vs " m))
              (is (< (Math/abs (- sd s)) 0.05) (str "sd " sd " vs " s))))))
      (finally
        (await-cps (sp/close! session))
        (ctx/stop-context! root)))))

;; --- blocks under the particle methods ----------------------------------------

(defn- sampled-gaussian-block
  "The Gaussian block with `:sample` from its prior N(0, s0²) and, unless
  `density?` is false, that prior's density as `:sample-log-density`."
  [d density?]
  (let [b (gaussian-block d true)
        prior (fn [{:keys [s0]}] (dist/normal 0.0 s0))]
    (block/block (:description b)
                 (cond-> (assoc (:capabilities b)
                                :sample (fn [inputs] (vec (repeatedly d #(dist/draw (prior inputs))))))
                   density?
                   (assoc :sample-log-density
                          (fn [^doubles mu inputs]
                            (reduce + (map #(dist/logpdf (prior inputs) %) mu))))))))

(deftest particle-methods-weigh-block-draws-by-their-target
  (let [b (sampled-gaussian-block 1 true)
        model (fn [] (spin (first (sample (block/block-dist b gauss-inputs) :id :mu))))
        [[mean _]] (gauss-posterior gauss-inputs 1)]
    (doseq [[label make] [["importance sampling" #(infer/importance-sampling (model) 4000)]
                          ["SMC" #(infer/smc-infer (model) 4000)]]]
      (testing label
        (let [m (first (w-mean-sd identity (weighted-values (run-infer 3 make))))]
          (is (< (Math/abs (- m mean)) 0.05) (str label ": " m " vs " mean))))))
  (testing "a block without :sample-log-density is refused, not mis-weighted"
    (let [b (sampled-gaussian-block 1 false)
          model (fn [] (spin (first (sample (block/block-dist b gauss-inputs) :id :mu))))
          outcome (try (run-infer 3 #(infer/importance-sampling (model) 10))
                       (catch Throwable e e))]
      (is (instance? Throwable outcome))
      (is (some #(= ::block/no-sample-density (:type (ex-data %)))
                (take-while some? (iterate ex-cause outcome)))))))

(deftest within-gibbs-counts-every-move
  ;; a block and one free latent: two moves a step, the free one (a prior
  ;; proposal with nothing observed) always accepted
  (let [b (gaussian-block 1 true)
        root (ctx/create-execution-context)
        session (sp/open! root {:seed 12 :fork-opts {:systems :none}})
        await-cps (fn [op] (let [p (promise)]
                             (op #(deliver p [:ok %]) #(deliver p [:err %]))
                             (let [[k v] (deref p 60000 [:err (ex-info "timeout" {})])]
                               (if (= k :ok) v (throw v)))))]
    (try
      (let [model (binding [ec/*execution-context* root]
                    (spin [(sample (block/block-dist b gauss-inputs) :id :mu :init [0.0])
                           (sample (dist/normal 0.0 1.0) :id :z)]))
            t0 (await-cps (trace/run session model (itrace/policy {:init? true}) {:anchor? itrace/anchor?}))
            {:keys [moves accepted]} (await-cps (itrace/mh-chain t0 20 {:step (hmc/within-gibbs {:step-size 0.1 :steps 8})}))]
        (is (= 40 moves))
        (is (< 20 accepted 41) (str accepted " of " moves)))
      (finally
        (await-cps (sp/close! session))
        (ctx/stop-context! root)))))

(deftest a-replay-that-adds-a-site-keeps-the-chain-exact
  ;; x ~ N(0, 1) as a block; a site w ~ N(0, 0.1) exists only when x > 0.
  ;; w is drawn afresh when a move crosses into x > 0 and dropped when it
  ;; crosses back: P(x > 0) must stay ½
  (let [b (block/block {:block/id :x :block/latents [{:name :x :shape [1]}]}
                       {:log-density (fn [^doubles x _] (* -0.5 (aget x 0) (aget x 0)))
                        :value+grad (fn [^doubles x _] [(* -0.5 (aget x 0) (aget x 0)) (double-array [(- (aget x 0))])])})
        model #(spin (let [x (sample (block/block-dist b nil) :id :theta :init [0.5])]
                      (when (pos? (first x)) (sample (dist/normal 0.0 0.1) :id :w))
                      (first x)))
        wv (weighted-values
            (run-infer 12 #(infer/kernel-infer (model) (k/hmc-kernel 3000 {:step-size 0.5 :steps 4 :samples :all :burn 200}) 4 {})))
        p (reduce + (map (fn [[x w]] (if (pos? x) w 0.0)) wv))]
    (is (< (Math/abs (- p 0.5)) 0.04) (str "P(x > 0) " p))))
