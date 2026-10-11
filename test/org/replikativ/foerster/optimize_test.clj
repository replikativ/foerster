(ns org.replikativ.foerster.optimize-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.optimize :as opt]
            [org.replikativ.foerster.random :as random]))

(def ^:private m [1.0 -2.0 0.5])
(def ^:private P [[4.0 1.0 0.0] [1.0 3.0 0.5] [0.0 0.5 2.0]])

(def ^:private gaussian
  ;; exp(1.3 − ½ (θ−m)ᵀ P (θ−m)), unnormalized
  (block/block {:block/id :g :block/latents [{:name :x :shape [3]}]}
               {:log-density (fn [th _] (let [d (mapv - th m)]
                                          (+ 1.3 (* -0.5 (reduce + (for [i (range 3) j (range 3)]
                                                                     (* (d i) (get-in P [i j]) (d j))))))))
                :value+grad (fn [th _] (let [d (mapv - th m)
                                             pd (mapv (fn [row] (reduce + (map * row d))) P)]
                                         [(+ 1.3 (* -0.5 (reduce + (map * d pd)))) (double-array (map - pd))]))}))

(def ^:private gamma-3-2
  ;; Gamma(α = 3, β = 2) on a positive latent, unnormalized
  (block/block {:block/id :s :block/coordinates :constrained
                :block/latents [{:name :s :shape [] :support :positive}]}
               {:log-density (fn [^doubles th _] (let [s (aget th 0)] (- (* 2.0 (Math/log s)) (* 2.0 s))))
                :value+grad (fn [^doubles th _] (let [s (aget th 0)]
                                                  [(- (* 2.0 (Math/log s)) (* 2.0 s)) (double-array [(- (/ 2.0 s) 2.0)])]))}))

(defn- close? [a b tol] (< (Math/abs (- a b)) tol))

(deftest laplace-is-exact-for-a-gaussian
  (let [{:keys [theta covariance log-evidence converged?]}
        (opt/laplace (block/block-dist gaussian nil) {:init [0.0 0.0 0.0]})
        ;; numpy: inv(P), 1.3 + 3/2 log 2π − ½ log det P
        sigma [[0.27380952380952384 -0.09523809523809525 0.023809523809523808]
               [-0.09523809523809525 0.38095238095238093 -0.09523809523809523]
               [0.023809523809523808 -0.09523809523809523 0.5238095238095238]]]
    (is converged?)
    (is (every? true? (map #(close? %1 %2 1e-7) theta m)))
    (is (every? true? (map #(close? %1 %2 1e-7) (flatten covariance) (flatten sigma))))
    (is (close? 2.534554380752307 log-evidence 1e-7))))

(deftest map-estimate-of-a-constrained-latent
  (let [d (block/block-dist gamma-3-2 nil)]
    (testing "the mode of σ itself: (α − 1)/β"
      (is (close? 1.0 (:s (:latents (opt/map-estimate d {:init [0.0]}))) 1e-7)))
    (testing "with the Jacobian, the mode of log σ: α/β"
      (is (close? 1.5 (:s (:latents (opt/map-estimate d {:init [0.0] :jacobian? true}))) 1e-7)))))

(deftest laplace-draws-map-back-through-the-transform
  ;; in θ = log σ the approximation is N(log 1.5, 1/3): σ is log-normal,
  ;; E σ = 1.5·exp(1/6)
  (random/set-seed! 3)
  (let [{:keys [draw covariance]} (opt/laplace (block/block-dist gamma-3-2 nil) {:init [0.0]})
        ss (map :s (repeatedly 20000 draw))]
    (is (close? (/ 1.0 3.0) (ffirst covariance) 1e-6))
    (is (every? pos? ss))
    (is (close? (* 1.5 (Math/exp (/ 1.0 6.0))) (/ (reduce + ss) 20000) 0.03))))

(deftest a-saddle-is-refused
  (let [saddle (block/block {:block/id :h :block/latents [{:name :x :shape [2]}]}
                            {:log-density (fn [^doubles th _] (- (* (aget th 1) (aget th 1)) (* (aget th 0) (aget th 0))))
                             :value+grad (fn [^doubles th _] [(- (* (aget th 1) (aget th 1)) (* (aget th 0) (aget th 0)))
                                                              (double-array [(* -2.0 (aget th 0)) (* 2.0 (aget th 1))])])})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not negative definite"
                          (opt/laplace (block/block-dist saddle nil) {:init [0.0 0.0] :max-iterations 0})))))

(defn- mean-of [xs] (/ (reduce + xs) (count xs)))

(deftest pathfinder-on-a-gaussian
  ;; on a quadratic target the L-BFGS path ends at the mean with an inverse
  ;; Hessian estimate close to the covariance; a block without :sample and
  ;; without :init starts at 0
  (random/set-seed! 4)
  (let [{:keys [mean covariance draws elbo paths]} (opt/pathfinder (block/block-dist gaussian nil) {})
        sigma [[0.27380952380952384 -0.09523809523809525 0.023809523809523808]
               [-0.09523809523809525 0.38095238095238093 -0.09523809523809523]
               [0.023809523809523808 -0.09523809523809523 0.5238095238095238]]]
    (is (every? true? (map #(close? %1 %2 0.02) mean m)))
    (is (every? true? (map #(close? %1 %2 0.1) (flatten covariance) (flatten sigma))))
    (is (= 1000 (count draws)))
    (is (every? true? (map #(close? (mean-of (map % draws)) %2 0.06) [first second #(nth % 2)] m)))
    (testing "the ELBO is at most log Z = 2.5346, up to Monte Carlo error"
      (is (< elbo 2.6)))
    (is (= 4 (count paths)))))

(deftest pathfinder-resamples-towards-the-target
  ;; in θ = log σ the Gamma(3, 2) target is skewed: the Gaussian's draws
  ;; overshoot E σ = 1.5, the importance-resampled draws do not
  (random/set-seed! 3)
  (let [{:keys [draws draw]} (opt/pathfinder (block/block-dist gamma-3-2 nil) {:draws 4000})
        resampled (mean-of (map #(Math/exp (first %)) draws))
        gaussian (mean-of (map :s (repeatedly 4000 draw)))]
    (is (close? 1.5 resampled 0.05))
    (is (> gaussian 1.65))))
