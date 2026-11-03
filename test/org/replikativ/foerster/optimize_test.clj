(ns org.replikativ.foerster.optimize-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.inference.optimize :as opt]
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

(def ^:private sigma
  ;; P⁻¹, the target's covariance
  [[0.27380952380952384 -0.09523809523809525 0.023809523809523808]
   [-0.09523809523809525 0.38095238095238093 -0.09523809523809523]
   [0.023809523809523808 -0.09523809523809523 0.5238095238095238]])

(defn- resampled-se
  "Monte Carlo standard error of the mean of resampled `draws` for a
  coordinate of posterior sd `sd`: duplicates carry no information, and the
  importance weights cost more, so the effective size is taken as half the
  distinct draws."
  [draws sd]
  (/ sd (Math/sqrt (/ (count (set draws)) 2.0))))

(defn- pathfinder-on-a-gaussian-holds? [seed]
  ;; On a quadratic target the L-BFGS path ends at the mean. The best
  ;; Gaussian is picked by an ELBO estimate from E draws: an iterate whose
  ;; mean is off by δ posterior sds loses δ²/2 in ELBO, while its estimate's
  ;; noise has sd ≈ δ/√E, so it wins only if δ < 2z/√E: with z ≤ 3 its mean
  ;; is within 6/√E sds. The resampled draws' means are within 4 standard
  ;; errors of the exact mean. A block without :sample and :init starts at 0.
  (random/set-seed! seed)
  (let [elbo-draws 400
        {:keys [mean covariance draws elbo paths]}
        (opt/pathfinder (block/block-dist gaussian nil) {:draws 4000 :elbo-draws elbo-draws})
        sds (mapv #(Math/sqrt (get-in sigma [% %])) (range 3))]
    {:mean (every? true? (map (fn [x mu sd] (< (Math/abs (- x mu)) (* (/ 6.0 (Math/sqrt elbo-draws)) sd))) mean m sds))
     :covariance (every? true? (map #(close? %1 %2 0.1) (flatten covariance) (flatten sigma)))
     :draws (and (= 4000 (count draws))
                 (every? true? (map (fn [j mu sd] (< (Math/abs (- (mean-of (map #(nth % j) draws)) mu))
                                                     (* 4 (resampled-se draws sd))))
                                    (range 3) m sds)))
     ;; the ELBO is at most log Z = 2.5346, up to Monte Carlo error
     :elbo (< elbo 2.6)
     :paths (= 4 (count paths))}))

(deftest pathfinder-on-a-gaussian
  (doseq [seed [4 5]]
    (let [r (pathfinder-on-a-gaussian-holds? seed)]
      (is (every? true? (vals r)) (str "seed " seed ": " r)))))

(deftest pathfinder-resamples-towards-the-target
  ;; in θ = log σ the Gamma(3, 2) target is skewed, so no Gaussian in θ is
  ;; exact: the importance-resampled draws still have E σ = 1.5 (sd √3/2),
  ;; within Monte Carlo error. The Gaussian the ELBO picks is not checked
  ;; against 1.5 (it may over- or undershoot, by seed); `draw` maps its own
  ;; draws back, so their mean is its lognormal mean exp(μ + v/2)
  (random/set-seed! 3)
  (let [{:keys [draws draw mean covariance]} (opt/pathfinder (block/block-dist gamma-3-2 nil) {:draws 4000})
        sigmas (map #(Math/exp (first %)) draws)
        gaussian-draws (map :s (repeatedly 4000 draw))
        gaussian (mean-of gaussian-draws)
        gaussian-se (/ (Math/sqrt (/ (reduce + (map #(let [e (- % gaussian)] (* e e)) gaussian-draws)) 3999.0))
                       (Math/sqrt 4000.0))]
    (is (< (Math/abs (- 1.5 (mean-of sigmas))) (* 4 (resampled-se sigmas (/ (Math/sqrt 3.0) 2.0)))) (str (mean-of sigmas)))
    (is (< (Math/abs (- gaussian (Math/exp (+ (first mean) (/ (ffirst covariance) 2.0))))) (* 4 gaussian-se))
        (str gaussian " ± " gaussian-se))))

(deftest pathfinder-options
  (random/set-seed! 5)
  (testing "one path, one draw"
    (is (= 1 (count (:draws (opt/pathfinder (block/block-dist gaussian nil) {:paths 1 :draws 1}))))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":draws"
                        (opt/pathfinder (block/block-dist gaussian nil) {:draws 0}))))

(deftest pathfinder-refuses-draws-of-zero-density
  ;; the L-BFGS path sees the Gaussian, but every draw scores −∞ (a target
  ;; with hard exclusions the Gaussians miss)
  (random/set-seed! 6)
  (with-redefs [opt/log-target (constantly ##-Inf)]
    (doseq [opts [{:paths 1 :draws 1} {:paths 2 :draws 10}]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"zero target density"
                            (opt/pathfinder (block/block-dist gaussian nil) opts))))))

(deftest the-reported-log-density-is-the-one-maximized
  ;; Gamma(3, 1) in s: 2 log s − s, mode s = 2 with 2 log 2 − 2; in θ = log s
  ;; the target gains the Jacobian θ: 3θ − eᶿ, mode s = 3 with 3 log 3 − 3
  (let [gamma-3-1 (block/block {:block/id :s :block/coordinates :constrained
                                :block/latents [{:name :s :shape [] :support :positive}]}
                               {:log-density (fn [^doubles th _] (let [s (aget th 0)] (- (* 2.0 (Math/log s)) s)))
                                :value+grad (fn [^doubles th _] (let [s (aget th 0)]
                                                                  [(- (* 2.0 (Math/log s)) s) (double-array [(- (/ 2.0 s) 1.0)])]))})
        natural (opt/map-estimate (block/block-dist gamma-3-1 nil) {:init [0.3]})
        unconstrained (opt/map-estimate (block/block-dist gamma-3-1 nil) {:init [0.3] :jacobian? true})]
    (is (close? (:s (:latents natural)) 2.0 1e-6))
    (is (close? (:log-density natural) (- (* 2.0 (Math/log 2.0)) 2.0) 1e-9) (str natural))
    (is (close? (:s (:latents unconstrained)) 3.0 1e-6))
    (is (close? (:log-density unconstrained) (- (* 3.0 (Math/log 3.0)) 3.0) 1e-9) (str unconstrained))))

(deftest pathfinder-takes-gradients-only-along-the-l-bfgs-path
  ;; the ELBO estimates and the importance weights need the log density
  ;; only: every gradient is one L-BFGS asked for
  (random/set-seed! 7)
  (let [in-lbfgs (atom false) outside (atom 0) densities (atom 0)
        lbfgs @#'opt/lbfgs
        vg (block/capability gaussian :value+grad)
        ld (block/capability gaussian :log-density)
        counted (block/block {:block/id :g :block/latents [{:name :x :shape [3]}]}
                             {:log-density (fn [th inputs] (swap! densities inc) (ld th inputs))
                              :value+grad (fn [th inputs] (when-not @in-lbfgs (swap! outside inc)) (vg th inputs))})]
    (with-redefs-fn {#'opt/lbfgs (fn [& args]
                                   (reset! in-lbfgs true)
                                   (try (apply lbfgs args) (finally (reset! in-lbfgs false))))}
      #(opt/pathfinder (block/block-dist counted nil) {:paths 2 :draws 100}))
    (is (zero? @outside))
    ;; 30 ELBO draws per candidate iterate, and the 100 proposals
    (is (>= @densities (+ 100 30)))))

;; -----------------------------------------------------------------------------
;; Pathfinder's diagonal-plus-low-rank inverse Hessian against the dense BFGS
;; reconstruction it replaced
;; -----------------------------------------------------------------------------

(defn- dot* [a b] (reduce + 0.0 (map * a b)))

(defn- dense-bfgs
  "The BFGS inverse updates by `pairs` [s y ρ] (newest last) applied in
  order to γI, γ = sᵀy/yᵀy of the newest pair, as a dense matrix: the
  previous Pathfinder's inverse Hessian."
  [pairs n]
  (let [[s y] (peek pairs) gamma (/ (dot* s y) (dot* y y))]
    (reduce (fn [h [s y rho]]
              (let [hy (mapv #(dot* % y) h) c (+ (* rho rho (dot* y hy)) rho)]
                (vec (for [i (range n)]
                       (vec (for [j (range n)]
                              (+ (get-in h [i j]) (* (- rho) (+ (* (nth s i) (nth hy j)) (* (nth hy i) (nth s j))))
                                 (* c (nth s i) (nth s j)))))))))
            (vec (for [i (range n)] (vec (for [j (range n)] (if (= i j) gamma 0.0)))))
            pairs)))

(defn- dense-approximation
  "The previous Pathfinder Gaussian at an iterate: dense BFGS from γI,
  drawn by its Cholesky factor."
  [theta grad pairs _alpha]
  (let [n (count theta) h (dense-bfgs pairs n)
        mean (mapv - theta (mapv #(dot* % grad) h))]
    (when-let [l (#'opt/cholesky h)]
      (let [log-det (* 2.0 (reduce + 0.0 (map #(Math/log (get-in l [% %])) (range n))))
            at (fn [z] [(mapv (fn [i] (+ (nth mean i) (reduce + 0.0 (map #(* (get-in l [i %]) (nth z %)) (range (inc i))))))
                              (range n))
                        (- (* -0.5 (dot* z z)) (* 0.5 log-det) (* n 0.9189385332046727))])]
        {:mean mean :covariance (delay h) :at at
         :draw #(at (vec (repeatedly n (fn [] (dist/draw (dist/normal 0.0 1.0))))))}))))

(defn- curvature-pairs
  "`m` pairs (s, y = A·s) of a random SPD A in dimension d."
  [^java.util.Random rng d m]
  (let [a (vec (repeatedly d (fn [] (vec (repeatedly d #(.nextGaussian rng))))))
        spd (vec (for [i (range d)]
                   (vec (for [j (range d)] (+ (if (= i j) 1.0 0.0) (dot* (a i) (a j)))))))]
    (vec (repeatedly m #(let [s (vec (repeatedly d (fn [] (.nextGaussian rng))))
                              y (mapv (fn [row] (dot* row s)) spd)]
                          [s y (/ 1.0 (dot* s y))])))))

(defn- max-abs-diff [a b] (apply max (map #(Math/abs (- %1 %2)) (flatten a) (flatten b))))

(deftest the-factored-inverse-hessian-is-the-dense-bfgs-one
  ;; with the diagonal γI the compact form is exactly the BFGS updates; Alg.
  ;; 4's draws have covariance H and its log q is the Gaussian's density —
  ;; for d above and below 2m
  (let [rng (java.util.Random. 3)]
    (doseq [[d m] [[7 4] [3 6] [20 6]]]
      (let [pairs (curvature-pairs rng d m)
            [s y] (peek pairs)
            factors (#'opt/inverse-hessian-factors pairs (vec (repeat d (/ (dot* s y) (dot* y y)))))
            h (dense-bfgs pairs d)
            mean (vec (repeatedly d #(.nextGaussian rng)))
            q (#'opt/gaussian mean factors)
            columns (for [i (range d)] (mapv - (first ((:at q) (assoc (vec (repeat d 0.0)) i 1.0))) mean))
            cov (vec (for [i (range d)] (vec (for [j (range d)] (reduce + (map #(* (nth % i) (nth % j)) columns))))))
            z (vec (repeatedly d #(.nextGaussian rng)))
            [x lq] ((:at q) z)
            dense-lq (second ((:at (dense-approximation (vec (repeat d 0.0)) (vec (repeat d 0.0)) pairs nil))
                              ;; the same point under the dense Cholesky factor
                              (let [l (#'opt/cholesky h)
                                    li (#'opt/lower-inverse l)]
                                (mapv #(dot* % (mapv - x mean)) li))))]
        (is (< (max-abs-diff (#'opt/dense-inverse-hessian factors) h) 1e-10) (str d " " m))
        (is (< (max-abs-diff cov h) 1e-10) (str d " " m))
        (is (< (max-abs-diff (#'opt/factors-times factors z) (mapv #(dot* % z) h)) 1e-10))
        (is (< (Math/abs (- lq dense-lq)) 1e-9) (str lq " vs " dense-lq))))))

(defn- draw-moments [draws j]
  (let [xs (map #(nth % j) draws) n (count xs) mean (/ (reduce + xs) n)]
    [mean (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) xs)) (dec n)))]))

(deftest pathfinder-draws-agree-with-the-dense-version
  ;; with :diagonal :lbfgs, the same approximations, drawn differently: the
  ;; resampled draws' means agree within Monte Carlo error, taking the
  ;; distinct draws as the effective sample size; both against the target's
  ;; exact means too
  (doseq [[label d exact] [["a correlated Gaussian" (block/block-dist gaussian nil) m]
                           ["Gamma(3, 2) in log σ" (block/block-dist gamma-3-2 nil) nil]]]
    (let [run (fn [approximation seed]
                (random/set-seed! seed)
                (with-redefs-fn {#'opt/approximation approximation}
                  #(:draws (opt/pathfinder d {:draws 4000 :diagonal :lbfgs}))))
          factored (run @#'opt/approximation 21)
          dense (run dense-approximation 22)]
      (doseq [j (range (count (first factored)))]
        (let [[m1 sd1] (draw-moments factored j) [m2 sd2] (draw-moments dense j)
              n1 (count (set factored)) n2 (count (set dense))
              se (Math/sqrt (+ (/ (* sd1 sd1) n1) (/ (* sd2 sd2) n2)))]
          (is (< (Math/abs (- m1 m2)) (* 4 se)) (str label " " j ": " m1 " vs " m2 " ± " se))
          (when exact
            (is (< (Math/abs (- m1 (nth exact j))) (* 4 (/ sd1 (Math/sqrt n1)))) (str label " " j))))))))

(deftest pathfinder-in-a-hundred-dimensions
  ;; N(μ, diag s²) with s within e^±0.3: L-BFGS reaches μ, and the factored
  ;; inverse Hessian of its last iterate is still the dense BFGS one
  (let [d 100
        s (mapv #(Math/exp (* 0.3 (Math/sin (* 0.37 %)))) (range d))
        mu (mapv #(Math/cos (* 0.21 %)) (range d))
        lp (fn [th] (* -0.5 (reduce + (map (fn [x m s] (let [z (/ (- x m) s)] (* z z))) th mu s))))
        b (block/block {:block/id :g100 :block/latents [{:name :x :shape [d]}]}
                       {:log-density (fn [th _] (lp th))
                        :value+grad (fn [th _] [(lp th) (double-array (map (fn [x m s] (- (/ (- x m) (* s s)))) th mu s))])})
        _ (random/set-seed! 12)
        result (opt/pathfinder (block/block-dist b nil) {:draws 200})
        {:keys [path]} (#'opt/lbfgs (#'opt/objective (block/block-dist b nil) true) (vec (repeat d 0.5))
                                    {:record? true :history 6})
        {:keys [pairs]} (peek path)
        [s1 y1] (peek pairs)]
    (is (every? #(< (Math/abs %) 0.05) (map (fn [a m s] (/ (- a m) s)) (:mean result) mu s)))
    (is (= 200 (count (:draws result))))
    (is (< (max-abs-diff (#'opt/dense-inverse-hessian
                          (#'opt/inverse-hessian-factors pairs (vec (repeat d (/ (dot* s1 y1) (dot* y1 y1))))))
                         (dense-bfgs pairs d))
           1e-10))))

(deftest pathfinder-can-keep-the-l-bfgs-diagonal
  ;; :diagonal :lbfgs (L-BFGS's γI) against the exact means
  (random/set-seed! 23)
  (let [{:keys [draws]} (opt/pathfinder (block/block-dist gaussian nil) {:draws 4000 :diagonal :lbfgs})]
    (doseq [j (range 3)]
      (let [sd (Math/sqrt (get-in sigma [j j]))]
        (is (< (Math/abs (- (mean-of (map #(nth % j) draws)) (nth m j))) (* 4 (resampled-se draws sd))) (str j)))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":diagonal"
                        (opt/pathfinder (block/block-dist gaussian nil) {:diagonal :dense}))))
