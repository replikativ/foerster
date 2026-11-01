(ns org.replikativ.foerster.nuts-test
  "NUTS with adaptation on block sites, and constrained latents, against
  exact posteriors."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.nuts :as nuts]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private scales [0.05 0.2 0.5 1.0 2.0 3.0 5.0 8.0 12.0 20.0])

(def ^:private scaled-gaussian
  ;; independent N(0, sᵢ²) over four orders of magnitude: only an adapted
  ;; metric samples it well
  (let [lp #(reduce + (map (fn [x s] (* -0.5 (/ (* x x) (* s s)))) (vec %) scales))]
    (block/block {:block/id :scaled :block/target :complete-conditional
                  :block/latents [{:name :x :shape [10] :support :real}]}
                 {:log-density (fn [th _] (lp th))
                  :value+grad (fn [th _] [(lp th) (double-array (map (fn [x s] (- (/ x (* s s)))) (vec th) scales))])})))

(deftest nuts-adapts-to-an-ill-conditioned-target
  (let [measure (b/run-infer 121 #(infer/infer (spin (sample (block/block-dist scaled-gaussian nil) :id :x
                                                             :init (vec (repeat 10 0.5))))
                                               {:method :nuts :iterations 1000 :chains 4 :burn 500}))]
    (doseq [j (range 10)]
      (let [cs (d/chains measure #(nth % j))]
        (is (< (d/rhat cs) 1.01) (str j))
        (is (< (Math/abs (- 1.0 (/ (:sd (d/summary measure #(nth % j))) (nth scales j)))) 0.1) (str j))))))

(deftest the-fisher-metric-is-exact-for-a-gaussian
  ;; with g = −q/s², √(Var q / Var g) = s² for any sample
  (let [rng (java.util.Random. 5)
        qs (vec (repeatedly 30 (fn [] (mapv #(* % (.nextGaussian rng)) scales))))
        gs (mapv (fn [q] (mapv (fn [x s] (- (/ x (* s s)))) q scales)) qs)]
    (is (every? true? (map #(< (Math/abs (- 1.0 (/ %1 (* %2 %2)))) 1e-9)
                           (@#'nuts/fisher-variance qs gs) scales)))))

(deftest nuts-adapts-a-fisher-metric
  (let [measure (b/run-infer 122 #(infer/infer (spin (sample (block/block-dist scaled-gaussian nil) :id :x
                                                             :init (vec (repeat 10 0.5))))
                                               {:method :nuts :iterations 1000 :chains 4 :burn 500 :metric :fisher}))]
    (doseq [j (range 10)]
      (let [cs (d/chains measure #(nth % j))]
        (is (< (d/rhat cs) 1.01) (str j))
        (is (< (Math/abs (- 1.0 (/ (:sd (d/summary measure #(nth % j))) (nth scales j)))) 0.1) (str j)))))
  (is (thrown? AssertionError (k/nuts-kernel 100 {:metric :dense}))))

(def ^:private ys [2.1 3.4 1.7 2.9 2.5 3.8 2.2 3.1])

(defn- normal-lp+grad
  "mu ~ N(0, 10), sigma ~ half-normal(2), ys ~ N(mu, sigma), in (mu, sigma)."
  [^doubles x]
  (let [mu (aget x 0) s (aget x 1)]
    [(+ (* -0.5 (/ (* mu mu) 100.0)) (- (* 0.5 (/ (* s s) 4.0)))
        (reduce + (map #(- (* -0.5 (/ (* (- % mu) (- % mu)) (* s s))) (Math/log s)) ys)))
     (double-array [(+ (- (/ mu 100.0)) (reduce + (map #(/ (- % mu) (* s s)) ys)))
                    (+ (- (/ s 4.0)) (reduce + (map #(- (/ (* (- % mu) (- % mu)) (* s s s)) (/ 1.0 s)) ys)))])]))

(def ^:private mean-and-scale
  (block/block {:block/id :ms :block/coordinates :constrained :block/target :complete-conditional
                :block/latents [{:name :mu :shape [] :support :real}
                                {:name :sigma :shape [] :support :positive}]}
               {:log-density (fn [x _] (first (normal-lp+grad x)))
                :value+grad (fn [x _] (normal-lp+grad x))}))

(def ^:private grid-means
  (delay
    (let [pts (for [mu (range 0.0 5.0 0.01) s (range 0.005 4.0 0.01)]
                [mu s (first (normal-lp+grad (double-array [mu s])))])
          top (apply max (map peek pts))
          ws (map #(Math/exp (- (peek %) top)) pts)
          z (reduce + ws)]
      [(/ (reduce + (map * (map first pts) ws)) z) (/ (reduce + (map * (map second pts) ws)) z)])))

(deftest constrained-latents-in-natural-coordinates
  (testing "σ > 0 by its log"
    (let [measure (b/run-infer 122 #(infer/infer (spin (sample (block/block-dist mean-and-scale nil) :id :th :init [2.0 0.0]))
                                                 {:method :nuts :iterations 1000 :chains 4 :burn 500}))
          [mu sigma] @grid-means
          s (fn [i] (d/summary measure #(nth (block/constrain mean-and-scale %) i)))]
      (is (< (Math/abs (- (:mean (s 0)) mu)) (* 4 (:mcse (s 0)))) (str (s 0) " vs " mu))
      (is (< (Math/abs (- (:mean (s 1)) sigma)) (* 4 (:mcse (s 1)))) (str (s 1) " vs " sigma))))
  (testing "p in (0, 1) by its logit: Beta(2, 2), 7 of 10 → Beta(9, 5)"
    (let [b (block/block {:block/id :p :block/coordinates :constrained :block/target :complete-conditional
                          :block/latents [{:name :p :shape [] :support [:interval 0.0 1.0]}]}
                         {:log-density (fn [x _] (let [p (aget ^doubles x 0)] (+ (* 8 (Math/log p)) (* 4 (Math/log (- 1 p))))))
                          :value+grad (fn [x _] (let [p (aget ^doubles x 0)]
                                                  [(+ (* 8 (Math/log p)) (* 4 (Math/log (- 1 p))))
                                                   (double-array [(- (/ 8 p) (/ 4 (- 1 p)))])]))})
          measure (b/run-infer 123 #(infer/infer (spin (sample (block/block-dist b nil) :id :p :init [0.0]))
                                                 {:method :nuts :iterations 1000 :chains 4 :burn 500}))
          s (d/summary measure #(first (block/constrain b %)))]
      (is (< (Math/abs (- (:mean s) (/ 9.0 14))) (* 4 (:mcse s))) (str s))
      (is (< (Math/abs (- (:sd s) (Math/sqrt (/ (* 9 5) (* 14 14 15.0))))) 0.01) (str s)))))

(deftest an-incomplete-target-is-refused
  ;; the block covers mu's prior only; the observation of mu is outside it
  (let [b (block/block {:block/id :prior-only :block/target :complete-conditional
                        :block/latents [{:name :mu :shape [] :support :real}]}
                       {:log-density (fn [th _] (* -0.5 (let [m (aget ^doubles th 0)] (* m m))))
                        :value+grad (fn [th _] (let [m (aget ^doubles th 0)] [(* -0.5 m m) (double-array [(- m)])]))})]
    (is (= ::nuts/incomplete-target
           (try (b/run-infer 124 #(infer/infer (spin (let [[mu] (sample (block/block-dist b nil) :id :mu :init [0.0])]
                                                       (observe (dist/normal mu 1.0) 2.0 :id :y)
                                                       mu))
                                               {:method :nuts :iterations 20 :chains 1 :burn 10}))
                nil
                (catch Exception e
                  (loop [e e] (cond (nil? e) nil
                                    (= ::nuts/incomplete-target (:type (ex-data e))) ::nuts/incomplete-target
                                    :else (recur (ex-cause e))))))))))

(deftest a-numeric-gradient-is-an-explicit-opt-in
  ;; the μ/σ block from its density alone, gradient by central differences
  (let [b (block/block {:block/id :ms-numeric :block/coordinates :constrained :block/target :complete-conditional
                        :block/latents [{:name :mu :shape [] :support :real}
                                        {:name :sigma :shape [] :support :positive}]}
                       (block/with-numeric-gradient {:log-density (fn [x _] (first (normal-lp+grad x)))}))
        x (double-array [2.3 -0.4])
        [_ g] ((block/capability b :value+grad) x nil)
        [_ g'] ((block/capability mean-and-scale :value+grad) x nil)]
    (is (every? #(< (Math/abs %) 1e-5) (map - g g')))))

(deftest an-interval-transform-is-exact-in-its-tails
  ;; p ~ Uniform(0, 1) in natural coordinates: in θ = logit p the density is
  ;; σ(θ)(1 − σ(θ)) ≈ e^−|θ|, also where σ rounds to 1
  (let [b (block/block {:block/id :p :block/coordinates :constrained
                        :block/latents [{:name :p :shape [] :support [:interval 0.0 1.0]}]}
                       {:log-density (fn [_ _] 0.0)
                        :value+grad (fn [_ _] [0.0 (double-array [0.0])])})
        d (block/block-dist b nil)
        [lp [g]] (block/value+grad d [40.0])]
    (is (< (Math/abs (- lp -40.0)) 1e-9) (str lp))
    (is (< (Math/abs (- g -1.0)) 1e-9) (str g))))

(deftest explicit-seeds-and-portable-sampler-statistics
  (let [target (block/block {:block/id :seeded :block/target :complete-conditional
                             :block/latents [{:name :x :shape [] :support :real}]}
                            {:log-density (fn [x _] (* -0.5 (aget x 0) (aget x 0)))
                             :value+grad (fn [x _] [(* -0.5 (aget x 0) (aget x 0))
                                                    (double-array [(- (aget x 0))])])})
        run (fn [process-seed seed]
              (b/run-infer process-seed
                           #(infer/infer (spin (sample (block/block-dist target nil) :id :x :init [0.0]))
                                         {:method :nuts :seed seed :iterations 30 :burn 10 :chains 2 :max-depth 2})))
        a (run 12 731) b (run 999 731) other (run 12 732)
        values #(mapv (comp measure/get-value first) (measure/get-particles %))
        stats (mapv (comp :sampler-stats first) (measure/get-particles a))]
    (is (= (values a) (values b)))
    (is (not= (values a) (values other)))
    (is (= 731 (:seed a)))
    (is (= [20 20] (:chain-lengths a)))
    (is (= 40 (count stats)))
    (is (every? boolean? (map :diverging stats)))
    (is (every? pos? (map :n-leapfrog stats)))
    (is (every? #(<= 1 (:tree-depth %) 2) stats))
    (is (every? #(<= 0.0 (:accept-stat %) 1.0) stats))
    (is (every? #(= (:reached-max-depth? %) (= 2 (:tree-depth %))) stats))
    (is (every? #(= 1 (:blocks %)) stats))))

(deftest divergent-transitions-remain-visible-in-samples
  (let [target (block/block {:block/id :narrow :block/target :complete-conditional
                             :block/latents [{:name :x :shape [] :support :real}]}
                            {:log-density (fn [x _] (* -5e11 (aget x 0) (aget x 0)))
                             :value+grad (fn [x _] [(* -5e11 (aget x 0) (aget x 0))
                                                    (double-array [(* -1e12 (aget x 0))])])})
        result (b/run-infer 21 #(infer/infer (spin (sample (block/block-dist target nil) :id :x :init [0.0]))
                                             {:method :nuts :seed 123 :iterations 4 :burn 0 :chains 2 :max-depth 2}))]
    (is (some (comp :diverging :sampler-stats first) (measure/get-particles result)))))

(defn- counting-gaussian
  "A standard normal block of dimension `d` that counts its density and
  gradient evaluations in the atoms `densities` and `gradients`."
  [d densities gradients]
  (let [lp (fn [^doubles th] (* -0.5 (areduce th i acc 0.0 (+ acc (* (aget th i) (aget th i))))))]
    (block/block {:block/id :counted :block/target :complete-conditional
                  :block/latents [{:name :x :shape [d] :support :real}]}
                 {:log-density (fn [th _] (swap! densities inc) (lp th))
                  :value+grad (fn [^doubles th _] (swap! gradients inc)
                                [(lp th) (double-array (map - th))])})))

(deftest a-transition-reuses-the-previous-gradient
  ;; every gradient is a leapfrog step's, except the first transition's start
  ;; point and the check of the jittered start. The trace scores a block by
  ;; :log-density only: once for the initial state, once for the jittered
  ;; start's replay, and once per transition that moved (its write-back)
  (doseq [[jitter extra-gradients extra-densities] [[false 1 1] [true 2 2]]]
    (let [densities (atom 0) gradients (atom 0)
          b (counting-gaussian 3 densities gradients)
          result (b/run-infer 31 #(infer/infer (spin (sample (block/block-dist b nil) :id :x :init [0.0 0.0 0.0]))
                                               {:method :nuts :seed 1 :iterations 200 :burn 0 :chains 1 :jitter jitter}))
          particles (measure/get-particles result)
          stats (map (comp :sampler-stats first) particles)
          values (map (comp measure/get-value first) particles)
          leapfrogs (reduce + (map :n-leapfrog stats))
          ;; a transition that moved: its draw differs from the one before
          ;; (the first draw always differs from the start here)
          moved (inc (count (filter true? (map not= (rest values) values))))]
      (is (= 200 (count stats)))
      (is (= (+ leapfrogs extra-gradients) @gradients) (str "jitter " jitter))
      (is (= (+ moved extra-densities) @densities) (str "jitter " jitter)))))

(deftest an-invalid-subtree-does-not-count-as-depth
  ;; Stan's base_nuts increments the depth only for a valid subtree: a tree
  ;; that ends at depth d has 2^d − 1 leapfrog steps when it ends by a U-turn
  ;; or the cap, and between 2^d and 2^(d+1) − 1 when its last subtree was
  ;; invalid — a divergence at the first step is depth 0
  (let [narrow (block/block {:block/id :narrow :block/target :complete-conditional
                             :block/latents [{:name :x :shape [] :support :real}]}
                            {:log-density (fn [x _] (* -5e3 (aget x 0) (aget x 0)))
                             :value+grad (fn [x _] [(* -5e3 (aget x 0) (aget x 0))
                                                    (double-array [(* -1e4 (aget x 0))])])})
        result (b/run-infer 22 #(infer/infer (spin (sample (block/block-dist narrow nil) :id :x :init [0.01]))
                                             {:method :nuts :seed 7 :iterations 50 :burn 0 :chains 2 :max-depth 6}))
        stats (map (comp :sampler-stats first) (measure/get-particles result))]
    (is (some :diverging stats))
    (doseq [{:keys [tree-depth n-leapfrog diverging]} stats]
      (is (<= (dec (bit-shift-left 1 tree-depth)) n-leapfrog (dec (bit-shift-left 1 (inc tree-depth))))
          (str tree-depth " " n-leapfrog))
      (when (and diverging (= 1 n-leapfrog))
        (is (zero? tree-depth))))))

(defn- isotropic-gaussian
  "N(0, s²I) in dimension d."
  [d s]
  (let [lp (fn [^doubles th] (* -0.5 (/ (areduce th i a 0.0 (+ a (* (aget th i) (aget th i)))) (* s s))))]
    (block/block {:block/id :iso :block/target :complete-conditional
                  :block/latents [{:name :x :shape [d] :support :real}]}
                 {:log-density (fn [th _] (lp th))
                  :value+grad (fn [^doubles th _] [(lp th) (double-array (map #(- (/ % (* s s))) th))])})))

(deftest the-step-size-heuristic-finds-the-targets-scale
  ;; Stan's init_stepsize from ε = 0.1 on N(0, s²I): the step it settles on
  ;; follows the scale over six orders of magnitude (leapfrog on a Gaussian
  ;; is stable below ε = 2s)
  (let [init @#'nuts/init-step-size]
    (doseq [s [1e-3 1.0 1e3] d [1 10]]
      (let [dd (block/block-dist (isotropic-gaussian d s) nil)
            q (vec (repeat d (* 0.5 s)))
            eps (for [seed (range 50)]
                  (random/with-stream* [seed 1] ::heuristic
                    #(init dd (repeat d 1.0) q (block/value+grad! dd (double-array q)) 0.1)))]
        (is (every? #(<= 0.25 (/ % s) 8.0) eps) (str s " " d " " (vec eps)))))
    (testing "an extreme step size is left alone, as Stan does"
      (let [dd (block/block-dist (isotropic-gaussian 1 1.0) nil)]
        (is (= 2e7 (init dd [1.0] [0.0] [0.0 (double-array [0.0])] 2e7)))))
    (testing "a flat target is improper: every step is accepted"
      (let [flat (block/block {:block/id :flat :block/latents [{:name :x :shape [] :support :real}]}
                              {:log-density (fn [_ _] 0.0) :value+grad (fn [_ _] [0.0 (double-array [0.0])])})]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"improper"
                              (init (block/block-dist flat nil) [1.0] [0.0] [0.0 (double-array [0.0])] 0.1)))))))

(deftest the-step-size-heuristic-starts-warm-up-and-every-metric-window
  ;; warm-up 200: Stan's windows [75, 100) and [100, 150); the heuristic runs
  ;; at the start and after each, and dual averaging restarts at μ = log 10ε
  ;; from the step size it found
  (let [heuristic (atom []) restarts (atom [])
        da-init @#'nuts/da-init]
    (with-redefs-fn {#'nuts/init-step-size (fn [_ _ _ _ eps] (swap! heuristic conj eps) 0.25)
                     #'nuts/da-init (fn [eps] (swap! restarts conj eps) (da-init eps))}
      #(b/run-infer 41 (fn [] (infer/infer (spin (sample (block/block-dist (isotropic-gaussian 2 1.0) nil) :id :x
                                                         :init [0.1 0.2]))
                                           {:method :nuts :seed 3 :iterations 210 :burn 200 :chains 1}))))
    (is (= 3 (count @heuristic)))
    (is (= 0.1 (first @heuristic)))
    ;; the first state's dual averaging, then one restart per heuristic
    (is (= [0.1 0.25 0.25 0.25] @restarts))))

(deftest adapted-tuning-starts-a-frozen-run
  ;; warm-up adapts ε and the diagonal metric; its retained draws report
  ;; them, and a second run given them with :burn 0 runs frozen at them,
  ;; resumed from the first run's last state (no jitter: with :burn 0 every
  ;; draw is kept, so a start must already be a posterior draw)
  (let [model #(spin (sample (block/block-dist scaled-gaussian nil) :id :x :init %))
        warm (b/run-infer 125 #(infer/infer (model (vec (repeat 10 0.5)))
                                            {:method :nuts :seed 11 :iterations 600 :chains 1 :burn 500}))
        particles (measure/get-particles warm)
        stats (mapv (comp :sampler-stats first) particles)
        last-draw (measure/get-value (first (peek particles)))
        {:keys [step-size inverse-metric]} (peek stats)]
    (is (= 1 (count (set (map :step-size stats)))) "fixed after warm-up")
    (is (= #{:x} (set (keys step-size)) (set (keys inverse-metric))))
    ;; the variance estimate of N(0, s²) coordinates is s²
    (doseq [[v s] (map vector (:x inverse-metric) scales)]
      (is (< 0.5 (/ v (* s s)) 2.0) (str v " vs " (* s s))))
    (let [frozen (b/run-infer 126 #(infer/infer (model last-draw)
                                                {:method :nuts :seed 12 :iterations 1000 :chains 4 :burn 0 :jitter false
                                                 :step-size step-size :inverse-metric inverse-metric}))
          fstats (mapv (comp :sampler-stats first) (measure/get-particles frozen))]
      (is (every? #(= step-size (:step-size %)) fstats))
      (is (every? #(= inverse-metric (:inverse-metric %)) fstats))
      (doseq [j (range 10)]
        (let [cs (d/chains frozen #(nth % j))]
          (is (< (d/rhat cs) 1.01) (str j))
          (is (< (Math/abs (- 1.0 (/ (:sd (d/summary frozen #(nth % j))) (nth scales j)))) 0.1) (str j)))))
    (testing "frozen NUTS is a fixed kernel a population can move by"
      (is (not (transition/adaptation? (transition/prepare (k/nuts-kernel 1 {:step-size step-size
                                                                             :inverse-metric inverse-metric}))))))))

(deftest injected-tuning-is-validated
  (is (thrown? AssertionError (k/nuts-kernel 10 {:step-size 0.0})))
  (is (thrown? AssertionError (k/nuts-kernel 10 {:inverse-metric [1.0 -1.0]})))
  (is (thrown? AssertionError (k/nuts-kernel 10 {:step-size {:x ##Inf}})))
  (is (= ::nuts/inverse-metric-dimension
         (try (b/run-infer 127 #(infer/infer (spin (sample (block/block-dist (isotropic-gaussian 2 1.0) nil) :id :x :init [0.0 0.0]))
                                             {:method :nuts :iterations 5 :chains 1 :inverse-metric [1.0 2.0 3.0]}))
              nil
              (catch Exception e
                (loop [e e] (cond (nil? e) nil
                                  (:type (ex-data e)) (if (= ::nuts/inverse-metric-dimension (:type (ex-data e)))
                                                        (:type (ex-data e)) (recur (ex-cause e)))
                                  :else (recur (ex-cause e)))))))))

(defn- starts
  "Each chain's first draw under NUTS frozen at a step size so small that it
  is the chain's start to 1e-9: `opts` go to infer."
  [seed block init opts]
  (let [result (b/run-infer seed #(infer/infer (spin (sample (block/block-dist block nil) :id :x :init init))
                                               (merge {:method :nuts :iterations 1 :burn 0 :max-depth 1
                                                       :step-size 1e-12}
                                                      opts)))]
    (mapv (comp measure/get-value first) (measure/get-particles result))))

(deftest chains-start-jittered-around-their-init
  (let [init [0.5 -1.0]
        target (isotropic-gaussian 2 1.0)
        jittered (starts 51 target init {:chains 40 :seed 5})
        offsets (mapcat #(map - % init) jittered)]
    (testing "Stan's radius: uniform in (−2, 2) per coordinate of θ"
      (is (every? #(< -2.0 % 2.0) offsets))
      (is (= 80 (count (set offsets))) "every chain and coordinate apart")
      ;; 80 draws of U(−2, 2): mean 0 (sd 0.13), variance 4/3 (sd ≈ 0.12)
      (let [mean (/ (reduce + offsets) 80)
            variance (/ (reduce + (map #(* (- % mean) (- % mean)) offsets)) 79)]
        (is (< (Math/abs mean) 0.45) (str mean))
        (is (< (Math/abs (- variance (/ 4.0 3.0))) 0.45) (str variance))))
    (testing "keyed to the chain's stream: the same seed starts the same"
      (is (= jittered (starts 52 target init {:chains 40 :seed 5})))
      (is (not= jittered (starts 51 target init {:chains 40 :seed 6}))))
    (testing ":jitter false starts at :init"
      (is (every? #(every? (fn [[x x0]] (< (Math/abs (- x x0)) 1e-9)) (map vector % init))
                  (starts 53 target init {:chains 4 :seed 5 :jitter false}))))
    (testing "a site no jitter can leave keeps its :init"
      (let [pinned (block/block {:block/id :pinned :block/target :complete-conditional
                                 :block/latents [{:name :x :shape [2] :support :real}]}
                                {:log-density (fn [^doubles x _] (if (zero? (aget x 0)) 0.0 ##-Inf))
                                 :value+grad (fn [^doubles x _] [(if (zero? (aget x 0)) 0.0 ##-Inf)
                                                                 (double-array 2)])})]
        (is (every? #(< (Math/abs (- (first %) 0.0)) 1e-9)
                    (starts 54 pinned [0.0 0.0] {:chains 2 :seed 5})))))))

(defn- dependent-blocks
  "a ~ N(0, 1) as block :a; b ~ N(a, 1), y = 2 ~ N(b, 1) as block :b, whose
  inputs are a: b's density depends on a, so a's block is not a complete
  conditional."
  []
  (let [a-block (block/block {:block/id :a :block/target :complete-conditional
                              :block/latents [{:name :a :shape [] :support :real}]}
                             {:log-density (fn [^doubles th _] (* -0.5 (aget th 0) (aget th 0)))
                              :value+grad (fn [^doubles th _] [(* -0.5 (aget th 0) (aget th 0)) (double-array [(- (aget th 0))])])})
        b-lp (fn [b a] (+ (* -0.5 (- b a) (- b a)) (* -0.5 (- 2.0 b) (- 2.0 b))))
        b-block (block/block {:block/id :b :block/target :complete-conditional
                              :block/latents [{:name :b :shape [] :support :real}]}
                             {:log-density (fn [^doubles th {:keys [a]}] (b-lp (aget th 0) a))
                              :value+grad (fn [^doubles th {:keys [a]}]
                                            (let [b (aget th 0)]
                                              [(b-lp b a) (double-array [(+ (- a b) (- 2.0 b))])]))})]
    (spin (let [[a] (sample (block/block-dist a-block nil) :id :a :init [0.0])
                [b] (sample (block/block-dist b-block {:a a}) :id :b :init [0.0])]
            [a b]))))

(deftest a-block-other-sites-depend-on-is-refused-with-the-workarounds
  (let [e (try (b/run-infer 128 #(infer/infer (dependent-blocks) {:method :nuts :iterations 5 :chains 1 :burn 0}))
               nil
               (catch Exception e
                 (loop [e e] (cond (nil? e) nil
                                   (= ::nuts/incomplete-target (:type (ex-data e))) e
                                   :else (recur (ex-cause e))))))]
    (is (some? e))
    (is (= :a (:address (ex-data e))))
    (is (= [:b] (:changed (ex-data e))))
    (is (re-find #"complete conditional" (ex-message e)))
    (is (re-find #"same block" (ex-message e)))
    (is (re-find #"k/hmc-kernel" (ex-message e)))
    (is (re-find #"k/delayed-acceptance-kernel" (ex-message e))))
  (testing "HMC, a suggested workaround, samples the same model exactly"
    ;; a ~ N(0,1), b | a ~ N(a,1), 2 | b ~ N(b,1): E[a | y] = y/3, E[b | y] = 2y/3
    (let [measure (b/run-infer 129 #(infer/kernel-infer (dependent-blocks)
                                                        (k/hmc-kernel 4000 {:step-size 0.3 :steps 5 :samples :all :burn 500})
                                                        4))
          s (fn [i] (d/summary measure #(nth % i)))]
      (is (< (Math/abs (- (:mean (s 0)) (/ 2.0 3.0))) (* 4 (:mcse (s 0)))) (str (s 0)))
      (is (< (Math/abs (- (:mean (s 1)) (/ 4.0 3.0))) (* 4 (:mcse (s 1)))) (str (s 1))))))

(defn- log-phi [x] (* -0.5 x x))

(deftest nuts-samples-known-laws-through-every-new-transform
  ;; one block, its latents independent, each a law with known moments:
  ;; lo = 3 + Exp(2) by [:lower 3]; up = −1 − Exp(1) by [:upper −1];
  ;; o, the order statistics of two N(0, 1), by :ordered (E = ∓1/√π);
  ;; p ~ Dirichlet(2, 3, 5) by :simplex (E = αₖ/10)
  (let [alpha [2.0 3.0 5.0]
        lp (fn [^doubles x]
             (let [lo (aget x 0) up (aget x 1) o1 (aget x 2) o2 (aget x 3)]
               (+ (* -2.0 (- lo 3.0)) (- (- -1.0 up)) (log-phi o1) (log-phi o2)
                  (reduce + (map (fn [a k] (* (- a 1.0) (Math/log (aget x (+ 4 k))))) alpha (range 3))))))
        grad (fn [^doubles x]
               (double-array (concat [-2.0 1.0 (- (aget x 2)) (- (aget x 3))]
                                     (map (fn [a k] (/ (- a 1.0) (aget x (+ 4 k)))) alpha (range 3)))))
        b (block/block {:block/id :laws :block/coordinates :constrained :block/target :complete-conditional
                        :block/latents [{:name :lo :shape [] :support [:lower 3.0]}
                                        {:name :up :shape [] :support [:upper -1.0]}
                                        {:name :o :shape [2] :support :ordered}
                                        {:name :p :shape [3] :support :simplex}]}
                       {:log-density (fn [x _] (lp x)) :value+grad (fn [x _] [(lp x) (grad x)])})
        measure (b/run-infer 131 #(infer/infer (spin (sample (block/block-dist b nil) :id :th :init (vec (repeat 6 0.0))))
                                               {:method :nuts :seed 9 :iterations 1500 :chains 4 :burn 500}))
        x (fn [i] (d/summary measure #(nth (block/constrain b %) i)))
        dirichlet-sd (fn [a] (Math/sqrt (/ (* a (- 10.0 a)) (* 100.0 11.0))))
        expected [[3.5 0.5] [-2.0 1.0] [(- (/ 1.0 (Math/sqrt Math/PI))) nil] [(/ 1.0 (Math/sqrt Math/PI)) nil]
                  [0.2 (dirichlet-sd 2.0)] [0.3 (dirichlet-sd 3.0)] [0.5 (dirichlet-sd 5.0)]]]
    ;; a rare divergence in a Dirichlet's tails, not a systematic one
    (is (< (count (filter (comp :diverging :sampler-stats first) (measure/get-particles measure))) 30))
    (doseq [[i [mean sd]] (map-indexed vector expected)]
      (let [s (x i)]
        (is (< (d/rhat (d/chains measure #(nth (block/constrain b %) i))) 1.01) (str i))
        (is (< (Math/abs (- (:mean s) mean)) (* 4 (:mcse s))) (str i " " s " vs " mean))
        (when sd
          (is (< (Math/abs (- 1.0 (/ (:sd s) sd))) 0.1) (str i " " s " vs sd " sd)))))))

(deftest the-trace-scores-a-block-by-its-log-density
  ;; z ~ Bernoulli(½) picks σ_z ∈ {0.5, 2}; μ ~ N(0, 1), y = 2 ~ N(μ, σ_z), μ a
  ;; block with inputs {:s σ_z}. Its :value+grad drops −log σ_z, a constant in
  ;; θ that NUTS never sees but MH on z must: the trace scores the block by
  ;; :log-density, so the posterior is exact (P(z = 1) = 0.624, E μ = 0.851)
  (let [y 2.0 sigmas [0.5 2.0]
        lp (fn [m s] (+ (* -0.5 m m) (* -0.5 (/ (* (- y m) (- y m)) (* s s))) (- (Math/log s))))
        sloppy (block/block {:block/id :mu :block/target :complete-conditional
                             :block/latents [{:name :mu :shape [] :support :real}]}
                            {:log-density (fn [^doubles th {:keys [s]}] (lp (aget th 0) s))
                             :value+grad (fn [^doubles th {:keys [s]}]
                                           (let [m (aget th 0)]
                                             [(+ (lp m s) (Math/log s))
                                              (double-array [(+ (- m) (/ (- y m) (* s s)))])]))})
        weights (mapv (fn [s] (/ (Math/exp (/ (* -0.5 y y) (+ 1 (* s s)))) (Math/sqrt (+ 1 (* s s))))) sigmas)
        p-z1 (/ (weights 1) (reduce + weights))
        e-mu (+ (* (- 1 p-z1) (/ y (+ 1 (* 0.25)))) (* p-z1 (/ y (+ 1 4.0))))
        measure (b/run-infer 132 #(infer/infer (spin (let [z (sample (dist/bernoulli 0.5) :id :z :init 0)
                                                           [mu] (sample (block/block-dist sloppy {:s (nth sigmas z)})
                                                                        :id :mu :init [0.0])]
                                                       [z mu]))
                                               {:method :nuts :seed 13 :iterations 3000 :burn 500 :chains 4}))
        z (d/summary measure #(double (first %)))
        mu (d/summary measure second)]
    (is (< (Math/abs (- p-z1 0.624)) 1e-3))
    (is (< (Math/abs (- (:mean z) p-z1)) (* 4 (:mcse z))) (str z " vs " p-z1))
    (is (< (Math/abs (- (:mean mu) e-mu)) (* 4 (:mcse mu))) (str mu " vs " e-mu))))

