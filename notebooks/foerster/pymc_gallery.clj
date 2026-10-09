;; # From the PyMC Gallery

;; Models from [PyMC's example gallery](https://www.pymc.io/projects/examples/en/latest/),
;; written in foerster and checked as the classic [gallery](foerster.gallery.html)
;; is: against an answer computed without sampling where one exists — here
;; a posterior on a grid — and otherwise against PyMC's published results.
;; Every number below is computed as the notebook renders, with fixed seeds.

(ns foerster.pymc-gallery
  (:require [clojure.string :as str]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as diagnostics]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

(def world (sp/create-execution-context))

(defn run [seed f]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        out (sp/with-context world @(f))]
    {:measure out :ms (long (/ (- (System/nanoTime) t0) 1e6))}))

(defn r3 [x] (/ (Math/round (* 1000.0 (double x))) 1000.0))
(defn r4 [x] (/ (Math/round (* 10000.0 (double x))) 10000.0))

;; ## 1. Golf putting
;;
;; Andrew Gelman's [case study](https://mc-stan.org/users/documentation/case-studies/golf.html),
;; in PyMC's [putting workflow](https://www.pymc.io/projects/examples/en/latest/case_studies/putting_workflow.html):
;; professional golfers' putts at distances from 2 to 20 feet, how many
;; were tried and how many went in (Berry 1995).

(def golf
  {:distance [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
   :tries [1443 694 455 353 272 256 240 217 200 237 202 192 174 167 201 195 191 147 152]
   :successes [1346 577 337 208 149 136 111 69 67 75 52 46 54 28 27 31 33 20 24]})

;; **A logistic regression** says the log-odds of success fall linearly
;; with distance, with Normal(0, 1) priors on both coefficients as in PyMC.
;; `binomial-logit` takes the log-odds itself: a prior draw like b = 2 puts
;; them near 40 at 20 feet, where σ rounds p to 1.0 and a plain `binomial`
;; would call every miss impossible.

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn logistic-golf []
  (spin
   (let [a (sample (dist/normal 0.0 1.0) :id :a)
         b (sample (dist/normal 0.0 1.0) :id :b)]
     (loop [i 0]
       (when (< i 19)
         (observe (dist/binomial-logit (nth (:tries golf) i) (+ a (* b (nth (:distance golf) i))))
                  (nth (:successes golf) i) :id [:putt i])
         (recur (inc i))))
     {:a a :b b})))

;; **A geometric model** says instead that a putt goes in when its angle
;; error is small enough for the ball to fit the cup: with ball radius r and
;; cup radius R, the threshold angle at distance x is asin((R − r)/x), and
;; with a Normal(0, σ) angle error the probability is 2Φ(threshold/σ) − 1.
;; One parameter, σ ~ HalfNormal(1):

(def ball-radius (/ (/ 1.68 2) 12))
(def cup-radius (/ (/ 4.25 2) 12))

(defn p-in [sigma x]
  (- (* 2.0 (dist/normal-cdf (/ (Math/asin (/ (- cup-radius ball-radius) x)) sigma))) 1.0))

(defn geometric-golf []
  (spin
   (let [sigma (sample (dist/half-normal 1.0) :id :sigma :init 0.1)]
     (loop [i 0]
       (when (< i 19)
         (observe (dist/binomial (nth (:tries golf) i) (p-in sigma (nth (:distance golf) i)))
                  (nth (:successes golf) i) :id [:putt i])
         (recur (inc i))))
     {:sigma sigma})))

;; Both posteriors are concentrated, so random-walk Metropolis–Hastings needs
;; steps on their scale, four chains each. The logistic model's two
;; coefficients are strongly correlated (the distances are not centred),
;; which a random walk crosses slowly: it gets long chains. The geometric
;; chains start at σ = 0.1, far above the posterior, and walk down during the
;; warm-up:

(def logistic-run
  (run 1 #(infer/infer (logistic-golf) {:method :rmh :iterations 20000 :burn 2000 :chains 4 :step-size 0.02})))

(def geometric-run
  (run 2 #(infer/infer (geometric-golf) {:method :rmh :iterations 4000 :burn 1000 :chains 4 :step-size 0.0006})))

;; **The exact posteriors** follow from the same densities on a grid: two
;; dimensions for the logistic model, one for the geometric.

(defn log-lik [p-of]
  (reduce + (map (fn [n y x] (dist/logpdf (dist/binomial n (p-of x)) y))
                 (:tries golf) (:successes golf) (:distance golf))))

(defn grid-moments
  "Posterior means and sds of the coordinates on a grid: `axes` a vector of
  [lo hi steps], `log-post` of a point (vector)."
  [axes log-post]
  (let [points (reduce (fn [ps [lo hi k]]
                         (for [p ps i (range k)] (conj p (+ lo (* i (/ (- hi lo) (dec k)))))))
                       [[]] axes)
        lps (mapv log-post points)
        top (reduce max lps)
        ws (mapv #(Math/exp (- % top)) lps)
        total (reduce + ws)
        moment (fn [f] (/ (reduce + (map #(* %1 (f %2)) ws points)) total))]
    (vec (for [j (range (count axes))]
           (let [mu (moment #(nth % j))]
             [mu (Math/sqrt (- (moment #(let [v (nth % j)] (* v v))) (* mu mu)))])))))

(def logistic-exact
  (grid-moments [[1.9 2.55 161] [-0.31 -0.21 161]]
                (fn [[a b]] (+ (dist/logpdf (dist/normal 0.0 1.0) a) (dist/logpdf (dist/normal 0.0 1.0) b)
                               (log-lik #(sigmoid (+ a (* b %))))))))

(def geometric-exact
  (grid-moments [[0.024 0.0295 551]]
                (fn [[s]] (+ (dist/logpdf (dist/half-normal 1.0) s) (log-lik #(p-in s %))))))

(kind/table
 {:column-names ["parameter" "exact mean" "exact sd" "mean" "sd" "R-hat" "bulk ESS"]
  :row-vectors (for [[label run k [mu sd]] [["a (logistic)" logistic-run :a (first logistic-exact)]
                                            ["b (logistic)" logistic-run :b (second logistic-exact)]
                                            ["σ (geometric)" geometric-run :sigma (first geometric-exact)]]]
                 (let [s (diagnostics/summary (:measure run) k)]
                   [label (r4 mu) (r4 sd) (r4 (:mean s)) (r4 (:sd s)) (r3 (:rhat s)) (long (:ess-bulk s))]))})

;; Which model predicts better? PSIS-LOO compares their expected log
;; predictive density on a held-out distance (each distance's putts are one
;; observation):

(kind/table
 {:column-names ["model" "elpd-loo" "se" "p-loo" "elpd difference" "se of difference" "max Pareto k"]
  :row-vectors (for [{:keys [name elpd-loo se p-loo elpd-diff dse]}
                     (diagnostics/compare {"geometric" (:measure geometric-run) "logistic" (:measure logistic-run)})]
                 [name (r3 elpd-loo) (r3 se) (r3 p-loo) (r3 elpd-diff) (r3 dse)
                  (r3 (apply max (vals (:pareto-k (diagnostics/loo (:measure (if (= name "geometric") geometric-run logistic-run)))))))])})

;; Read the warnings before the ranking. A Pareto k̂ above 0.7 says leaving a
;; distance out moves the posterior too far for the importance-sampling
;; estimate to be trusted, and an effective number of parameters `p-loo`
;; far above the actual one (two, and one) says the model misfits the
;; observations it depends on most. Both are expected here: each distance
;; pools up to 1443 putts, so every observation is very informative. The
;; difference is over twice its standard error and the geometric model is
;; ahead at 15 of the 19 distances — at two feet alone by 52 — so the
;; ranking stands; its size should not be read precisely.

;; One physical parameter beats two free ones: the logistic curve cannot
;; follow the success rate's fall from near certainty at two feet. The fits
;; against the data:

(-> (tc/dataset (for [[i x] (map-indexed vector (:distance golf))
                      [label p] [["observed" (/ (double (nth (:successes golf) i)) (nth (:tries golf) i))]
                                 ["logistic" (let [[[a] [b]] logistic-exact] (sigmoid (+ a (* b x))))]
                                 ["geometric" (p-in (ffirst geometric-exact) x)]]]
                  {:distance x :p p :series label}))
    (plotly/layer-line {:=x :distance :=y :p :=color :series}))

;; ## 2. Radon in Minnesota homes
;;
;; PyMC's [multilevel modeling](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html)
;; example (from Gelman & Hill 2006): the log radon level of 919 homes in 85
;; counties, measured in the basement (floor 0) or on the first floor. The
;; varying-intercept model gives each county its own intercept, drawn from
;; a common normal — partial pooling: a county with few homes is pulled
;; toward the state mean, one with many keeps its own.
;;
;;     μ_a ~ N(0, 10),  σ_a ~ Exp(1),  α_c ~ N(μ_a, σ_a)   (85 counties)
;;     β ~ N(0, 10),    σ_y ~ Exp(1),  y_i ~ N(α_county(i) + β floor_i, σ_y)
;;
;; (The data: PyMC's `radon.csv`, MIT licence, reduced to the three columns
;; used here.)

(def radon
  (let [[_ & lines] (str/split-lines (slurp "notebooks/data/radon.csv"))
        rows (mapv #(let [[c f y] (str/split % #",")]
                      [(Long/parseLong c) (Long/parseLong f) (Double/parseDouble y)])
                   lines)]
    {:county (int-array (map first rows))
     :floor (double-array (map second rows))
     :y (double-array (map #(nth % 2) rows))
     :counties 85}))

;; Eighty-nine parameters: a block (see [blocks](foerster.blocks.html))
;; with its gradient written out, θ = [μ_a, σ_a, β, σ_y, α_1 … α_85] with
;; σ_a and σ_y positive, sampled by NUTS. The block also shows each home's
;; log density (`:pointwise`), for LOO.

(defn radon-value+grad [^doubles th {:keys [^ints county ^doubles floor ^doubles y counties]}]
  (let [mu-a (aget th 0) s-a (aget th 1) beta (aget th 2) s-y (aget th 3)
        n (alength y)
        g (double-array (alength th))
        lp (volatile! (+ (* -0.5 (/ (* mu-a mu-a) 100.0)) (- s-a)
                         (* -0.5 (/ (* beta beta) 100.0)) (- s-y)))]
    (aset g 0 (/ (- mu-a) 100.0)) (aset g 1 -1.0) (aset g 2 (/ (- beta) 100.0)) (aset g 3 -1.0)
    (dotimes [c counties]
      (let [a (aget th (+ 4 c)) z (/ (- a mu-a) s-a)]
        (vswap! lp + (- (* -0.5 z z) (Math/log s-a)))
        (aset g (+ 4 c) (- (aget g (+ 4 c)) (/ z s-a)))
        (aset g 0 (+ (aget g 0) (/ z s-a)))
        (aset g 1 (+ (aget g 1) (/ (- (* z z) 1.0) s-a)))))
    (dotimes [i n]
      (let [c (aget county i)
            r (/ (- (aget y i) (aget th (+ 4 c)) (* beta (aget floor i))) s-y)]
        (vswap! lp + (- (* -0.5 r r) (Math/log s-y)))
        (aset g (+ 4 c) (+ (aget g (+ 4 c)) (/ r s-y)))
        (aset g 2 (+ (aget g 2) (/ (* r (aget floor i)) s-y)))
        (aset g 3 (+ (aget g 3) (/ (- (* r r) 1.0) s-y)))))
    [(- @lp (* 0.5 n (Math/log (* 2 Math/PI))) (* 0.5 counties (Math/log (* 2 Math/PI)))) g]))

(def radon-block
  (block/block {:block/id :radon
                :block/coordinates :constrained
                :block/latents [{:name :mu-a :shape []}
                                {:name :sigma-a :shape [] :support :positive}
                                {:name :beta :shape []}
                                {:name :sigma-y :shape [] :support :positive}
                                {:name :alpha :shape [85]}]
                :block/target :complete-conditional}
               {:log-density (fn [th inputs] (first (radon-value+grad th inputs)))
                :value+grad radon-value+grad
                :pointwise (fn [^doubles th {:keys [^ints county ^doubles floor ^doubles y]}]
                             (into {} (for [i (range (alength y))]
                                        [i (dist/logpdf (dist/normal (+ (aget th (+ 4 (aget county i)))
                                                                        (* (aget th 2) (aget floor i)))
                                                                     (aget th 3))
                                                        (aget y i))])))}))

(defn radon-model []
  (spin (sample (block/block-dist radon-block radon) :id :theta :init (vec (repeat 89 0.0)))))

(def radon-run
  (run 3 #(infer/infer (radon-model) {:method :nuts :iterations 1500 :burn 500 :chains 4})))

;; **The exact posterior.** Given the two scales, everything else is linear
;; and Gaussian: within county c the n_c homes share α_c, so their
;; covariance is σ_y² I + σ_a² 11ᵀ, whose inverse and determinant are in
;; closed form; μ_a and β (normal priors) then integrate out exactly. That
;; leaves a two-dimensional posterior over (σ_a, σ_y), computed on a grid,
;; and the mean and variance of β given each grid point.

(defn radon-given-scales
  "[log p(y | σ_a, σ_y) up to a constant, posterior mean of (μ_a, β),
  posterior covariance of (μ_a, β)] at the scales."
  [s-a s-y {:keys [^ints county ^doubles floor ^doubles y counties]}]
  (let [n (alength y)
        sy2 (* s-y s-y) sa2 (* s-a s-a)
        ;; per county: n_c, Σ 1, Σ x, Σ y over its homes
        sums (reduce (fn [acc i]
                       (update acc (aget county i)
                               (fn [[k sx sy]] [(inc k) (+ sx (aget floor i)) (+ sy (aget y i))])))
                     (vec (repeat counties [0 0.0 0.0])) (range n))
        ;; V⁻¹ = (I − w 11ᵀ)/σ_y² per county, w = σ_a²/(σ_y² + n_c σ_a²)
        quad (fn [f g] ; f(i), g(i) → Σ_i Σ_j f_i (V⁻¹)_ij g_j
               (let [own (reduce + (map #(* (f %) (g %)) (range n)))
                     fs (reduce (fn [acc i] (update acc (aget county i) + (f i))) (vec (repeat counties 0.0)) (range n))
                     gs (reduce (fn [acc i] (update acc (aget county i) + (g i))) (vec (repeat counties 0.0)) (range n))]
                 (/ (- own (reduce + (map (fn [[k] a b] (* (/ sa2 (+ sy2 (* k sa2))) a b)) sums fs gs))) sy2)))
        one (constantly 1.0) x #(aget floor %) yy #(aget y %)
        a11 (+ (quad one one) 0.01) a12 (quad one x) a22 (+ (quad x x) 0.01)
        b1 (quad one yy) b2 (quad x yy)
        det (- (* a11 a22) (* a12 a12))
        m1 (/ (- (* a22 b1) (* a12 b2)) det) m2 (/ (- (* a11 b2) (* a12 b1)) det)
        log-det-v (reduce + (map (fn [[k]] (+ (* (dec k) (Math/log sy2)) (Math/log (+ sy2 (* k sa2))))) sums))]
    [(* -0.5 (+ (- (quad yy yy) (+ (* m1 b1) (* m2 b2))) log-det-v (Math/log det)))
     [m1 m2]
     [[(/ a22 det) (/ (- a12) det)] [(/ (- a12) det) (/ a11 det)]]]))

(def radon-exact
  (let [grid (for [s-a (map #(+ 0.15 (* 0.005 %)) (range 81))
                   s-y (map #(+ 0.66 (* 0.0025 %)) (range 61))]
               (let [[ll m c] (radon-given-scales s-a s-y radon)]
                 {:s-a s-a :s-y s-y :lp (- ll s-a s-y) :beta (second m) :var-beta (get-in c [1 1])}))
        top (reduce max (map :lp grid))
        ws (map #(Math/exp (- (:lp %) top)) grid)
        total (reduce + ws)
        e (fn [f] (/ (reduce + (map #(* %1 (f %2)) ws grid)) total))
        mean-sd (fn [f] (let [mu (e f)] [mu (Math/sqrt (- (e #(let [v (f %)] (* v v))) (* mu mu)))]))
        beta-mean (e :beta)]
    {:sigma-a (mean-sd :s-a) :sigma-y (mean-sd :s-y)
     ;; total variance: within a grid point, plus between
     :beta [beta-mean (Math/sqrt (+ (e :var-beta) (- (e #(* (:beta %) (:beta %))) (* beta-mean beta-mean))))]}))

(defn radon-natural [theta] (block/constrain radon-block theta))

(kind/table
 {:column-names ["parameter" "exact mean" "exact sd" "PyMC" "mean" "sd" "R-hat" "bulk ESS"]
  :row-vectors (for [[label k i pymc] [["β (floor)" :beta 2 "−0.664 ± 0.069"]
                                       ["σ_a" :sigma-a 1 ""] ["σ_y" :sigma-y 3 ""]]]
                 (let [s (diagnostics/summary (:measure radon-run) #(nth (radon-natural %) i))
                       [mu sd] (get radon-exact k)]
                   [label (r3 mu) (r3 sd) pymc (r3 (:mean s)) (r3 (:sd s)) (r3 (:rhat s)) (long (:ess-bulk s))]))})

;; Partial pooling at work: each county's intercept against its number of
;; homes. Counties with few homes sit near the state mean μ_a; with many,
;; the estimate follows the county's own data.

(let [draws (mapv (comp radon-natural m/get-value first) (m/get-particles (:measure radon-run)))
      homes (frequencies (seq (:county radon)))]
  (-> (tc/dataset (for [c (range 85)]
                    {:homes (get homes c 0)
                     :intercept (/ (reduce + (map #(nth % (+ 4 c)) draws)) (count draws))}))
      (plotly/layer-point {:=x :homes :=y :intercept})))

;; ## 3. Stochastic volatility
;;
;; PyMC's [stochastic volatility](https://www.pymc.io/projects/examples/en/latest/time_series/stochastic_volatility.html)
;; model: daily returns of the S&P 500 are Student-t with a scale e^(v_t)
;; whose logarithm v_t walks at random,
;;
;;     step ~ Exp(10),  ν ~ Exp(0.1),  v_0 ~ N(0, 100),  v_t ~ N(v_(t−1), step)
;;     r_t ~ StudentT(ν, 0, e^(v_t))
;;
;; Here over the last 300 trading days of PyMC's data (to 2019-11-14). PyMC
;; samples the whole volatility path with NUTS, and so does foerster, on a
;; block of 302 numbers. A random walk written as it reads — v_t given
;; v_(t−1) — gives NUTS a funnel: when the step is small the path must be
;; nearly straight, when it is large the path is free, and no single step
;; size fits both. The standard cure is to write the path *non-centred*:
;; sample the standardized innovations z_t ~ N(0, 1) and compute
;; v_t = v_(t−1) + step · z_t, which is the same model with a geometry NUTS
;; can cross.

(def days 300)

(def returns
  (let [[_ & lines] (str/split-lines (slurp "notebooks/data/sp500.csv"))]
    (double-array (map #(Double/parseDouble (second (str/split % #","))) (take-last days lines)))))

(defn digamma
  "ψ(x), x > 0: the recurrence up to 6, then the asymptotic series."
  [x]
  (loop [x (double x) acc 0.0]
    (if (< x 6.0)
      (recur (inc x) (- acc (/ 1.0 x)))
      (let [f (/ 1.0 (* x x))]
        (+ acc (Math/log x) (/ -0.5 x)
           (* f (+ (/ -1.0 12) (* f (+ (/ 1.0 120) (* f (+ (/ -1.0 252) (* f (+ (/ 1.0 240) (* f (/ -1.0 132)))))))))))))))

(defn sv-value+grad
  "θ = [step, ν, v_0, z_1 … z_(T−1)] (step and ν in their own coordinates):
  the path non-centred, v_t = v_(t−1) + step·z_t with z_t ~ N(0, 1), which
  removes the funnel between the step and the path."
  [^doubles th {:keys [^doubles r]}]
  (let [step (aget th 0) nu (aget th 1) n (alength r)
        g (double-array (alength th))
        v (double-array n)
        _ (aset v 0 (aget th 2))
        _ (loop [t 1] (when (< t n) (aset v t (+ (aget v (dec t)) (* step (aget th (+ 2 t))))) (recur (inc t))))
        c (- (dist/lgamma (* 0.5 (+ nu 1.0))) (dist/lgamma (* 0.5 nu)) (* 0.5 (Math/log (* nu Math/PI))))
        dc (- (* 0.5 (digamma (* 0.5 (+ nu 1.0)))) (* 0.5 (digamma (* 0.5 nu))) (/ 0.5 nu))
        gv (double-array n)                     ; ∂ log-lik / ∂v_t
        lp (volatile! (+ (Math/log 10.0) (* -10.0 step) (Math/log 0.1) (* -0.1 nu)
                         (* -0.5 (/ (* (aget v 0) (aget v 0)) 10000.0)) (- (Math/log 100.0)) -0.9189385332046727))]
    (aset g 0 -10.0) (aset g 1 -0.1)
    (dotimes [t n]
      (let [z (* (aget r t) (Math/exp (- (aget v t)))) q (/ (* z z) nu)]
        (vswap! lp + (- c (aget v t) (* 0.5 (+ nu 1.0) (Math/log1p q))))
        (aset gv t (+ -1.0 (/ (* (+ nu 1.0) q) (+ 1.0 q))))
        (aset g 1 (+ (aget g 1) dc (* -0.5 (Math/log1p q)) (/ (* 0.5 (+ nu 1.0) q) (* nu (+ 1.0 q)))))))
    ;; z_t moves v_t … v_(T−1): its gradient is step times the tail sum of ∂/∂v
    (loop [t (dec n) tail 0.0]
      (when (>= t 0)
        (let [tail (+ tail (aget gv t))]
          (if (zero? t)
            (aset g 2 (- tail (/ (aget v 0) 10000.0)))
            (let [zt (aget th (+ 2 t))]
              (vswap! lp + (- (* -0.5 zt zt) 0.9189385332046727))
              (aset g (+ 2 t) (- (* step tail) zt))
              (aset g 0 (+ (aget g 0) (* zt tail)))))
          (recur (dec t) tail))))
    [@lp g]))

(def sv-block
  (block/block {:block/id :sv
                :block/coordinates :constrained
                :block/latents [{:name :step :shape [] :support :positive}
                                {:name :nu :shape [] :support :positive}
                                {:name :v0 :shape []}
                                {:name :z :shape [(dec days)]}]
                :block/target :complete-conditional}
               {:log-density (fn [th inputs] (first (sv-value+grad th inputs)))
                :value+grad sv-value+grad}))

(defn sv-block-model []
  (spin (sample (block/block-dist sv-block {:r returns}) :id :theta
                :init (vec (concat [(Math/log 0.1) (Math/log 5.0) -4.5] (repeat (dec days) 0.0))))))

(def sv-nuts
  (run 4 #(infer/infer (sv-block-model) {:method :nuts :iterations 600 :burn 300 :chains 4})))

(defn sv-path
  "The volatility path e^(v_t) of a draw (θ in unconstrained coordinates)."
  [theta]
  (let [x (block/constrain sv-block theta) step (nth x 0)]
    (mapv #(Math/exp %) (reductions (fn [v z] (+ v (* step z))) (nth x 2) (subvec (vec x) 3)))))

(kind/table
 {:column-names ["parameter" "mean" "sd" "R-hat" "bulk ESS"]
  :row-vectors (for [[label f] [["step" #(nth (block/constrain sv-block %) 0)]
                                ["ν" #(nth (block/constrain sv-block %) 1)]
                                ["volatility, last day" #(peek (sv-path %))]]]
                 (let [s (diagnostics/summary (:measure sv-nuts) f)]
                   [label (r4 (:mean s)) (r4 (:sd s)) (r3 (:rhat s)) (long (:ess-bulk s))]))})

;; The returns and the posterior mean of the volatility e^(v_t), with ±2
;; times it as a band:

(let [paths (mapv (comp sv-path m/get-value first) (take-nth 10 (m/get-particles (:measure sv-nuts))))
      mean-path (apply mapv (fn [& vs] (/ (reduce + vs) (count vs))) paths)]
  (-> (tc/dataset (concat (for [t (range days)] {:day t :value (aget ^doubles returns t) :series "return"})
                          (for [t (range days)] {:day t :value (* 2 (nth mean-path t)) :series "+2 volatility"})
                          (for [t (range days)] {:day t :value (* -2 (nth mean-path t)) :series "−2 volatility"})))
      (plotly/layer-line {:=x :day :=y :value :=color :series})))

;; **A second algorithm on the last day.** On the final day, the smoothed
;; volatility NUTS reports and the filtered volatility a particle filter
;; reports are the same quantity, p(v_T | all returns). The same model
;; written with sites — the path a sequence of sample sites, each return an
;; observation — runs under SMC, here with the two parameters fixed at
;; their posterior means (NUTS also averages over them, so the two need not
;; agree exactly):

(defn sv-model [step nu]
  (spin
   (loop [t 0
          v (sample (dist/normal 0.0 100.0) :id [:v 0]
                    :proposal (dist/normal (Math/log (Math/abs (aget ^doubles returns 0))) 1.5))]
     (observe (dist/student-t nu 0.0 (Math/exp v)) (aget ^doubles returns t) :id [:r t])
     (if (= t (dec days))
       (Math/exp v)
       (recur (inc t) (sample (dist/normal v step) :id [:v (inc t)]))))))

(def sv-filter
  (let [s (fn [i] (:mean (diagnostics/summary (:measure sv-nuts) #(nth (block/constrain sv-block %) i))))]
    (run 5 #(infer/infer (sv-model (s 0) (s 1)) {:method :smc :particles 1000}))))

(kind/table
 {:column-names ["volatility on the last day" "mean" "sd"]
  :row-vectors [(let [s (diagnostics/summary (:measure sv-nuts) #(peek (sv-path %)))]
                  ["NUTS (smoothing, parameters integrated)" (r4 (:mean s)) (r4 (:sd s))])
                (let [s (diagnostics/summary (:measure sv-filter) identity)]
                  ["particle filter (parameters at their means)" (r4 (:mean s)) (r4 (:sd s))])]})

;; Two very different algorithms — gradient-based sampling of the whole
;; path, and a filter that never sees the gradient — agree on today's
;; volatility. The filter is also the model's online form: pushing each
;; new day's return into it (see [streaming](foerster.streaming.html))
;; updates today's volatility without refitting the year.
