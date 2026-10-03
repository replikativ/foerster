(ns org.replikativ.foerster.workflow-test
  "Checking a result: convergence diagnostics against ArviZ's numbers,
  chains kept apart by kernel-infer, posterior predictive draws, pointwise
  log-likelihoods, one call shape for every method, and the distributions
  standard models need."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- test-chains
  "`m` AR(1) chains of `n` draws (coefficient `rho`), the second shifted by
  `shift`, from java.util.Random — the same numbers ArviZ was given."
  [seed m n rho shift]
  (let [r (java.util.Random. seed)]
    (vec (for [c (range m)]
           (loop [t 0 x 0.0 out []]
             (if (= t n)
               out
               (let [x (+ (* rho x) (.nextGaussian r))]
                 (recur (inc t) x (conj out (+ x (if (= c 1) shift 0.0)))))))))))

(defn- close? [a b] (< (Math/abs (- a b)) (* 1e-9 (Math/abs a))))

(deftest diagnostics-agree-with-arviz
  ;; az.rhat, az.ess(method="bulk"/"tail"), az.mcse(method="mean"),
  ;; ArviZ 0.23.4, on the same chains
  (doseq [[label cs [rhat bulk tail mcse]]
          [["independent draws" (test-chains 1 4 200 0.0 0.0)
            [1.0027291943730958 850.4322165090186 799.1525412938777 0.033818517713052124]]
           ["autocorrelated" (test-chains 2 4 301 0.8 0.0)
            [1.0138655537194028 145.55278764854523 254.61120902177953 0.14074169290589492]]
           ["one chain elsewhere" (test-chains 3 3 150 0.0 1.0)
            [1.1234496432334309 20.13778738833606 280.9425604392309 0.25535721739560435]]]]
    (testing label
      (is (close? rhat (d/rhat cs)))
      (is (close? bulk (d/ess-bulk cs)))
      (is (close? tail (d/ess-tail cs)))
      (is (close? mcse (d/mcse cs))))))

(deftest diagnostics-of-constant-or-discrete-draws
  ;; ArviZ gives NaN where the draws carry no variation, not an error
  (let [stuck [[1.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0]]
        discrete (test-chains 4 4 400 0.0 0.0)
        discrete (mapv (fn [c] (mapv #(if (pos? %) 1.0 0.0) c)) discrete)]
    (is (Double/isNaN (d/rhat stuck)))
    (is (Double/isNaN (d/ess-bulk stuck)))
    (is (Double/isNaN (d/ess-tail stuck)))
    (is (< (d/rhat discrete) 1.02))
    (is (number? (d/ess-tail discrete)))))

(deftest diagnostics-of-long-chains-are-quick
  (let [cs (test-chains 5 4 3000 0.5 0.0)
        t0 (System/nanoTime)
        _ (doall [(d/rhat cs) (d/ess-bulk cs) (d/ess-tail cs) (d/mcse cs)])
        ms (/ (- (System/nanoTime) t0) 1e6)]
    (is (< ms 3000) (str ms " ms for 12000 draws"))))

(defn- gaussian []
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
     (observe (dist/normal mu 0.5) 1.0 :id :y1)
     (observe (dist/normal mu 0.5) 0.6 :id :y2)
     mu)))

;; mu | y ~ N(1.6/9 · 4 … ): precision 1 + 2·4 = 9, mean (4·1.6)/9
(def ^:private post-mean (/ 6.4 9.0))
(def ^:private post-sd (/ 1.0 3.0))

(deftest kernel-infer-keeps-its-chains
  (let [measure (b/run-infer 81 #(infer/infer (gaussian) {:method :rmh :iterations 1500 :chains 4
                                                          :burn 300 :step-size 0.6}))
        s (d/summary measure :mu)]
    (is (= [1200 1200 1200 1200] (:chain-lengths measure)))
    (is (= 4 (:chains s)))
    (is (< (:rhat s) 1.01) (str s))
    (is (> (:ess-bulk s) 400) (str s))
    (is (< (Math/abs (- (:mean s) post-mean)) (* 4 (:mcse s))) (str s))
    (testing "a chain started far off is flagged before it mixes"
      (let [short (b/run-infer 82 #(infer/kernel-infer
                                    (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu :init 8.0)]
                                            (observe (dist/normal mu 0.5) 1.0 :id :y1)
                                            mu))
                                    (org.replikativ.foerster.kernel/random-walk-mh-kernel 40 {:samples :all :step-size 0.05})
                                    4))]
        (is (> (:rhat (d/summary short :mu)) 1.01))))))

(deftest posterior-predictive-draws
  ;; y_new | data ~ N(post-mean, √(post-sd² + 0.5²))
  (let [measure (b/run-infer 83 #(infer/infer (gaussian) {:method :smc :particles 2000}))
        draws (b/run-infer 84 #(infer/predictive (gaussian) measure 3000))
        ys (map (comp :y1 :observations) draws)
        mu (/ (reduce + ys) (count ys))
        sd (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) ys)) (count ys)))]
    (is (= #{:y1 :y2} (set (keys (:observations (first draws))))))
    (is (< (Math/abs (- mu post-mean)) 0.05) (str mu))
    (is (< (Math/abs (- sd (Math/sqrt (+ (* post-sd post-sd) 0.25)))) 0.04) (str sd))
    (is (not= 1.0 (first ys)) "the observed site drew afresh")))

(deftest pointwise-log-likelihoods
  (let [measure (b/run-infer 85 #(infer/infer (gaussian) {:method :importance :particles 50}))
        ll (d/pointwise-log-likelihood measure)]
    (is (= 50 (count ll)))
    (is (every? #(= #{:y1 :y2} (set (keys %))) ll))
    (let [[particle] (first (m/get-particles measure))
          mu (m/get-value particle)]
      (is (< (Math/abs (- (dist/logpdf (dist/normal mu 0.5) 1.0) (:y1 (first ll)))) 1e-12)))))

(deftest one-call-shape-for-every-method
  (doseq [opts [{:method :importance :particles 3000}
                {:method :smc :particles 3000}
                {:method :tempered :particles 500}
                {:method :pimh :particles 200 :iterations 30}
                {:method :pgibbs :particles 50 :iterations 300}
                {:method :mh :iterations 1500 :chains 4 :burn 300}
                {:method :pmmh :particles 50 :iterations 400 :params #{:mu} :scale 0.5 :burn 50}]]
    (testing (name (:method opts))
      (let [measure (b/run-infer 86 #(infer/infer (gaussian) opts))
            {:keys [mean]} (d/summary measure identity)]
        (is (< (Math/abs (- mean post-mean)) 0.12) (str (:method opts) " " mean)))))
  (is (thrown? clojure.lang.ExceptionInfo (infer/infer (gaussian) {:method :nuts}))))

(deftest new-distributions
  (random/set-seed! 87)
  (let [n 20000
        mean-of (fn [d] (/ (reduce + (repeatedly n #(dist/draw d))) n))]
    (testing "mass sums to one"
      (is (< (Math/abs (- 1.0 (reduce + (map #(Math/exp (dist/logpdf (dist/binomial 12 0.3) %)) (range 13))))) 1e-12))
      (is (< (Math/abs (- 1.0 (reduce + (map #(Math/exp (dist/logpdf (dist/uniform-discrete 3 9) %)) (range 3 9))))) 1e-12)))
    (testing "draws have the stated means"
      (doseq [[label d tol] [["binomial" (dist/binomial 12 0.3) 0.05]
                             ["log-normal" (dist/log-normal 0.2 0.5) 0.03]
                             ["half-normal" (dist/half-normal 2.0) 0.03]
                             ["uniform-discrete" (dist/uniform-discrete 3 9) 0.05]]]
        (is (< (Math/abs (- (dist/mean d) (mean-of d))) tol) label)))
    (testing "densities integrate and quantiles invert the cdf"
      (doseq [[label d lo hi] [["log-normal" (dist/log-normal 0.2 0.5) 1e-6 30.0]
                               ["half-normal" (dist/half-normal 2.0) 0.0 30.0]
                               ["half-cauchy" (dist/half-cauchy 1.5) 0.0 2e4]
                               ["cauchy" (dist/cauchy 1.0 2.0) -2e4 2e4]]]
        (let [k 400000 h (/ (- hi lo) k)
              integral (* h (reduce + (map #(Math/exp (dist/logpdf d (+ lo (* h (+ % 0.5))))) (range k))))]
          (is (< (Math/abs (- 1.0 integral)) 2e-3) (str label " " integral)))
        (doseq [p [0.1 0.5 0.9]]
          (is (< (Math/abs (- p (dist/cdf d (dist/quantile d p)))) 1e-9) label))))
    (is (thrown? clojure.lang.ExceptionInfo (dist/binomial -1 0.5)))))

(deftest loo-and-waic-agree-with-arviz
  ;; az.loo / az.waic (ArviZ 0.23.4) of the same matrix, one chain, the last
  ;; observation an outlier (k̂ > 0.7)
  (let [S 1000
        ys [-0.4 0.1 0.3 0.8 1.2 -1.0 0.5 6.0]
        mu (mapv #(+ 0.2 (* 0.4 (Math/sin (* 1.7 %)) (Math/cos (* 0.31 %)))) (range S))
        sd (mapv #(+ 0.8 (* 0.3 (Math/abs (Math/sin (* 0.13 %))))) (range S))
        rows (mapv (fn [y] (mapv (fn [s] (- (* -0.5 (Math/pow (/ (- y (mu s)) (sd s)) 2))
                                            (Math/log (sd s)) (* 0.5 (Math/log (* 2 Math/PI)))))
                                 (range S)))
                   ys)
        l (#'d/loo-rows (vec (range 8)) rows {})
        w (#'d/waic-rows (vec (range 8)) rows)]
    (is (close? -33.26613819654955 (:elpd-loo l)))
    (is (close? 22.26593722334338 (:se l)))
    (is (close? 9.437383556995009 (:p-loo l)))
    (doseq [[i k] (map-indexed vector [-0.7504504140913563 -0.22560708703632104 -0.39163661138576306
                                       -0.3787971592024531 0.14113488898314375 -0.0671982732243114
                                       -0.6057390271064482 1.1188649155334143])]
      (is (close? k (get (:pareto-k l) i)) (str i)))
    (is (> (get (:pareto-k l) 7) (:good-k l)))
    (is (close? -37.606187534742496 (:elpd-waic w)))
    (is (close? 26.32456537721405 (:se w)))
    (is (close? 13.777432895187951 (:p-waic w)))))

(defn- located [mu-prior]
  (spin
   (let [mu (sample mu-prior :id :mu)]
     (loop [i 0]
       (when (< i 6)
         (observe (dist/normal mu 1.0) (nth [-0.4 0.1 0.3 0.8 1.2 -1.0] i) :id [:y i])
         (recur (inc i))))
     mu)))

(deftest loo-compares-models
  (let [fit #(b/run-infer 87 (fn [] (infer/infer (located %) {:method :mh :iterations 2000 :chains 2 :burn 200})))
        free (fit (dist/normal 0.0 3.0))
        pinned (fit (dist/normal 3.0 0.05))
        l (d/loo free)
        [best worse] (d/compare {:free free :pinned pinned})]
    (is (= 6 (count (:pointwise l))))
    (is (every? #(< % (:good-k l)) (vals (:pareto-k l))))
    (is (< (Math/abs (- (:elpd-loo l) (:elpd-waic (d/waic free)))) 0.2))
    (is (< 0.3 (:p-loo l) 1.5) "about one parameter")
    (is (= [:free :pinned] [(:name best) (:name worse)]))
    (is (zero? (:elpd-diff best)))
    (is (> (:elpd-diff worse) (* 2 (:dse worse))))))
(defn- line [xs ys]
  (spin
   (let [a (sample (dist/normal 0.0 5.0) :id :a)
         b (sample (dist/normal 0.0 5.0) :id :b)]
     (loop [i 0]
       (when (< i (count xs))
         (observe (dist/normal (+ a (* b (nth xs i))) 0.3) (nth ys i) :id [:y i])
         (recur (inc i))))
     [a b])))

(deftest predictive-on-new-inputs-and-under-interventions
  (let [xs [0.0 1.0 2.0 3.0 4.0] ys [1.1 2.9 5.2 6.9 9.1]
        measure (b/run-infer 88 #(infer/infer (line xs ys) {:method :mh :iterations 3000 :chains 2 :burn 500}))
        mean-of (fn [draws address] (let [vs (map #(get-in % [:observations address]) draws)]
                                      (/ (reduce + vs) (count vs))))
        new (b/run-infer 89 #(infer/predictive (line [10.0] [0.0]) measure 2000))
        done (b/run-infer 90 #(infer/predictive (line [10.0] [0.0]) measure 2000 {:interventions {:b 0.0}}))]
    (testing "a new input is predicted from the posterior, not its placeholder data"
      (is (= #{[:y 0]} (set (keys (:observations (first new))))))
      (is (< (Math/abs (- (mean-of new [:y 0]) 21.04)) 0.3) (str (mean-of new [:y 0]))))
    (testing "do(b = 0) leaves the intercept"
      (is (every? #(= 0.0 (second (:value %))) done))
      (is (< (Math/abs (- (mean-of done [:y 0]) 1.04)) 0.3) (str (mean-of done [:y 0]))))))
