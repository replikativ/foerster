(ns org.replikativ.foerster.dist
  "Probability distributions, portable between the JVM and JavaScript.

  A distribution is a value (a record) with

    (draw d)          a sample, from the current generator
                      (`foerster.random`: a world's stream at a site)
    (logpdf d x)      the log density, or log mass for a discrete law;
                      ##-Inf outside the support
    (cdf d x) (quantile d p) (mean d) (variance d)   where defined

  Names and parameterizations follow raster's distributions
  (`raster.sci.distributions`, and Distributions.jl): Normal[mu sigma],
  Uniform[a b], Exponential[lambda] (rate), Gamma[alpha beta] (shape, SCALE:
  mean αβ), Beta[alpha beta], Poisson[lambda]. A model's site laws and a
  raster block's compiled densities therefore mean the same thing."
  (:refer-clojure :exclude [flip])
  (:require [org.replikativ.foerster.random :as random]))

(defprotocol Distribution
  (-draw [d])
  (-logpdf [d x]))

(defprotocol Univariate
  (-cdf [d x])
  (-quantile [d p]))

(defprotocol Moments
  (-mean [d])
  (-variance [d]))

(defprotocol DrawDensity
  (-draw-logpdf [d x]
    "The log density of `x` under what `draw` samples, or nil when that is
    the distribution itself (`logpdf`)."))

(extend-protocol DrawDensity
  #?(:clj Object :cljs default)
  (-draw-logpdf [_ _] nil))

(defn distribution? [x] (satisfies? Distribution x))
(defn draw "A sample of `d`." [d] (-draw d))
(defn logpdf "The log density (log mass) of `d` at `x`." [d x] (-logpdf d x))
(defn draw-logpdf
  "The log density of `x` under what `(draw d)` samples: `logpdf`, unless `d`
  draws from something else (a block's `:sample`)."
  [d x]
  (or (-draw-logpdf d x) (-logpdf d x)))
(defn cdf [d x] (-cdf d x))
(defn quantile
  "The p-quantile of `d`, p in [0, 1]."
  [d p]
  (when-not (and (number? p) (<= 0.0 p 1.0))
    (throw (ex-info "A quantile's probability must lie in [0, 1]"
                    {:type ::invalid-probability :p p})))
  (-quantile d p))
(defn mean [d] (-mean d))
(defn variance [d] (-variance d))

;; =============================================================================
;; Special functions
;; =============================================================================

(def ^:private log-2pi (Math/log (* 2.0 Math/PI)))
(def ^:private log-sqrt-2pi (* 0.5 log-2pi))

(def ^:private lanczos
  [0.99999999999980993 676.5203681218851 -1259.1392167224028
   771.32342877765313 -176.61502916214059 12.507343278686905
   -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7])

(defn lgamma
  "log Γ(x) (Lanczos, g = 7): about 15 significant digits."
  [x]
  (if (< x 0.5)
    (- (Math/log (/ Math/PI (Math/abs (Math/sin (* Math/PI x)))))
       (lgamma (- 1.0 x)))
    (let [x (- x 1.0)
          a (reduce + (first lanczos)
                    (map-indexed (fn [i c] (/ c (+ x i 1.0))) (rest lanczos)))
          t (+ x 7.5)]
      (+ log-sqrt-2pi (* (+ x 0.5) (Math/log t)) (- t) (Math/log a)))))

(defn lbeta [a b] (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

(defn- poly
  "c0 + c1 r + c2 r² + … of `cs` = [c0 c1 …]."
  [cs r]
  (reduce (fn [acc c] (+ c (* r acc))) 0.0 (rseq cs)))

(defn normal-quantile
  "Φ⁻¹(p) (Wichura 1988, AS241 PPND16): about 16 significant digits."
  [p]
  (cond
    (<= p 0.0) ##-Inf
    (>= p 1.0) ##Inf
    :else
    (let [q (- p 0.5)]
      (if (<= (Math/abs q) 0.425)
        (let [r (- 0.180625 (* q q))]
          (/ (* q (poly [3.387132872796366608 133.14166789178437745
                         1971.5909503065514427 13731.693765509461125
                         45921.953931549871457 67265.770927008700853
                         33430.575583588128105 2509.0809287301226727] r))
             (poly [1.0 42.313330701600911252 687.1870074920579083
                    5394.1960214247511077 21213.794301586595867
                    39307.89580009271061 28729.085735721942674
                    5226.495278852545925] r)))
        (let [r (Math/sqrt (- (Math/log (if (neg? q) p (- 1.0 p)))))
              v (if (<= r 5.0)
                  (let [r (- r 1.6)]
                    (/ (poly [1.42343711074968357734 4.6303378461565452959
                              5.7694972214606914055 3.64784832476320460504
                              1.27045825245236838258 0.24178072517745061177
                              0.0227238449892691845833 7.7454501427834140764e-4] r)
                       (poly [1.0 2.05319162663775882187 1.6763848301838038494
                              0.68976733498510000455 0.14810397642748007459
                              0.0151986665636164571966 5.475938084995344946e-4
                              1.05075007164441684324e-9] r)))
                  (let [r (- r 5.0)]
                    (/ (poly [6.6579046435011037772 5.4637849111641143699
                              1.7848265399172913358 0.29656057182850489123
                              0.026532189526576123093 0.0012426609473880784386
                              2.71155556874348757815e-5 2.01033439929228813265e-7] r)
                       (poly [1.0 0.59983220655588793769 0.13692988092273580531
                              0.0148753612908506148525 7.868691311456132591e-4
                              1.8463183175100546818e-5 1.4215117583164458887e-7
                              2.04426310338993978564e-15] r))))]
          (if (neg? q) (- v) v))))))

(defn- gamma-series
  "P(a, x) by its series; for x < a + 1."
  [a x]
  (loop [n 1 term (/ 1.0 a) sum (/ 1.0 a)]
    (if (or (> n 1000) (< (Math/abs term) (* (Math/abs sum) 1e-17)))
      (* sum (Math/exp (- (* a (Math/log x)) x (lgamma a))))
      (let [term (* term (/ x (+ a n)))]
        (recur (inc n) term (+ sum term))))))

(defn- gamma-fraction
  "Q(a, x) by its continued fraction (Lentz); for x ≥ a + 1. Keeps its
  relative precision however small Q is."
  [a x]
  (let [tiny 1e-300]
    (loop [i 1 b (+ x 1.0 (- a)) c (/ 1.0 tiny) d (/ 1.0 (+ x 1.0 (- a)))
           h (/ 1.0 (+ x 1.0 (- a)))]
      (let [an (* (- i) (- i a))
            b (+ b 2.0)
            d (let [d (+ (* an d) b)] (/ 1.0 (if (< (Math/abs d) tiny) tiny d)))
            c (let [c (+ b (/ an c))] (if (< (Math/abs c) tiny) tiny c))
            delta (* d c)
            h (* h delta)]
        (if (or (> i 1000) (< (Math/abs (- delta 1.0)) 1e-17))
          (* h (Math/exp (- (* a (Math/log x)) x (lgamma a))))
          (recur (inc i) b c d h))))))

(defn regularized-gamma-p
  "P(a, x) = γ(a, x)/Γ(a), the regularized lower incomplete gamma function."
  [a x]
  (cond (<= x 0.0) 0.0
        (< x (+ a 1.0)) (gamma-series a x)
        :else (- 1.0 (gamma-fraction a x))))

(defn regularized-gamma-q
  "Q(a, x) = 1 − P(a, x), computed directly in the upper tail."
  [a x]
  (cond (<= x 0.0) 1.0
        (< x (+ a 1.0)) (- 1.0 (gamma-series a x))
        :else (gamma-fraction a x)))

(defn normal-cdf
  "Φ(z), the standard normal CDF: ½ Q(½, z²/2) below 0, which keeps its
  relative precision deep into the tail."
  [z]
  (let [tail (* 0.5 (regularized-gamma-q 0.5 (* 0.5 z z)))]
    (if (pos? z) (- 1.0 tail) tail)))

;; =============================================================================
;; Draws
;; =============================================================================

(defn- u01 [] (random/uniform01))

(defn- standard-normal
  "Box–Muller, one of the pair: two uniforms per draw, so a keyed stream
  draws the same numbers however the draws interleave."
  []
  (let [u1 (- 1.0 (u01))                ; (0, 1]
        u2 (u01)]
    (* (Math/sqrt (* -2.0 (Math/log u1))) (Math/cos (* 2.0 Math/PI u2)))))

(defn- standard-gamma
  "Gamma(shape, 1) (Marsaglia & Tsang 2000; shape < 1 by boosting)."
  [shape]
  (if (< shape 1.0)
    (* (standard-gamma (+ shape 1.0)) (Math/pow (- 1.0 (u01)) (/ 1.0 shape)))
    (let [d (- shape (/ 1.0 3.0))
          c (/ 1.0 (Math/sqrt (* 9.0 d)))]
      (loop []
        (let [x (standard-normal)
              v (+ 1.0 (* c x))]
          (if (<= v 0.0)
            (recur)
            (let [v (* v v v)
                  u (- 1.0 (u01))]
              (if (or (< u (- 1.0 (* 0.0331 x x x x)))
                      (< (Math/log u) (+ (* 0.5 x x) (* d (+ (- 1.0 v) (Math/log v))))))
                (* d v)
                (recur)))))))))

(defn- poisson-draw
  "Poisson(λ): Knuth's multiplication below 10, Hörmann's PTRS (1993) above."
  [lambda]
  (if (< lambda 10.0)
    (let [limit (Math/exp (- lambda))]
      (loop [k 0 p (u01)]
        (if (<= p limit) k (recur (inc k) (* p (u01))))))
    (let [slam (Math/sqrt lambda)
          loglam (Math/log lambda)
          b (+ 0.931 (* 2.53 slam))
          a (+ -0.059 (* 0.02483 b))
          invalpha (+ 1.1239 (/ 1.1328 (- b 3.4)))
          vr (- 0.9277 (/ 3.6224 (- b 2.0)))]
      (loop []
        (let [u (- (u01) 0.5)
              v (u01)
              us (- 0.5 (Math/abs u))
              k (Math/floor (+ (* (+ (/ (* 2.0 a) us) b) u) lambda 0.43))]
          (cond
            (and (>= us 0.07) (<= v vr)) (long k)
            (or (neg? k) (and (< us 0.013) (> v us))) (recur)
            (<= (+ (Math/log v) (Math/log invalpha) (- (Math/log (+ (/ a (* us us)) b))))
                (+ (- lambda) (* k loglam) (- (lgamma (+ k 1.0)))))
            (long k)
            :else (recur)))))))

;; =============================================================================
;; Distributions
;; =============================================================================

(defn- in? [lo x hi] (and (<= lo x) (<= x hi)))

(defn- xlogy
  "a·log x, 0 when a is 0 (so a flat factor x⁰ stays 1 at x = 0)."
  [a x]
  (if (zero? a) 0.0 (* a (Math/log x))))

(defn- whole?
  "A whole number, whatever its type: 3 and 3.0 alike, on both platforms."
  [x]
  (and (number? x) (== x (Math/floor x))))

(defrecord Normal [mu sigma]
  Distribution
  (-draw [_] (+ mu (* sigma (standard-normal))))
  (-logpdf [_ x]
    (let [z (/ (- x mu) sigma)]
      (- (* -0.5 z z) (Math/log sigma) log-sqrt-2pi)))
  Univariate
  (-cdf [_ x] (normal-cdf (/ (- x mu) sigma)))
  (-quantile [_ p] (+ mu (* sigma (normal-quantile p))))
  Moments
  (-mean [_] mu)
  (-variance [_] (* sigma sigma)))

(defrecord Uniform [a b]
  Distribution
  (-draw [_] (+ a (* (- b a) (u01))))
  (-logpdf [_ x] (if (in? a x b) (- (Math/log (- b a))) ##-Inf))
  Univariate
  (-cdf [_ x] (cond (< x a) 0.0 (> x b) 1.0 :else (/ (- x a) (- b a))))
  (-quantile [_ p] (+ a (* p (- b a))))
  Moments
  (-mean [_] (* 0.5 (+ a b)))
  (-variance [_] (/ (* (- b a) (- b a)) 12.0)))

(defrecord Exponential [lambda]
  Distribution
  (-draw [_] (/ (- (Math/log (- 1.0 (u01)))) lambda))
  (-logpdf [_ x] (if (neg? x) ##-Inf (- (Math/log lambda) (* lambda x))))
  Univariate
  (-cdf [_ x] (if (neg? x) 0.0 (- 1.0 (Math/exp (- (* lambda x))))))
  (-quantile [_ p] (/ (- (Math/log (- 1.0 p))) lambda))
  Moments
  (-mean [_] (/ 1.0 lambda))
  (-variance [_] (/ 1.0 (* lambda lambda))))

(defrecord Gamma [alpha beta]
  Distribution
  (-draw [_] (* beta (standard-gamma alpha)))
  (-logpdf [_ x]
    (if (<= x 0.0)
      ##-Inf
      (- (* (- alpha 1.0) (Math/log x)) (/ x beta) (lgamma alpha) (* alpha (Math/log beta)))))
  Univariate
  (-cdf [_ x] (regularized-gamma-p alpha (/ x beta)))
  (-quantile [_ _]
    (throw (ex-info "Gamma has no quantile yet" {:type ::unsupported})))
  Moments
  (-mean [_] (* alpha beta))
  (-variance [_] (* alpha beta beta)))

(defrecord Beta [alpha beta]
  Distribution
  (-draw [_]
    (let [x (standard-gamma alpha)
          y (standard-gamma beta)]
      (/ x (+ x y))))
  (-logpdf [_ x]
    (if (in? 0.0 x 1.0)
      (- (+ (xlogy (- alpha 1.0) x) (xlogy (- beta 1.0) (- 1.0 x)))
         (lbeta alpha beta))
      ##-Inf))
  Moments
  (-mean [_] (/ alpha (+ alpha beta)))
  (-variance [_] (/ (* alpha beta) (* (+ alpha beta) (+ alpha beta) (+ alpha beta 1.0)))))

(defrecord Poisson [lambda]
  Distribution
  (-draw [_] (poisson-draw lambda))
  (-logpdf [_ k]
    (if (and (whole? k) (not (neg? k)))
      (- (* k (Math/log lambda)) lambda (lgamma (+ k 1.0)))
      ##-Inf))
  Univariate
  (-cdf [_ x]
    (if (neg? x)
      0.0
      (- 1.0 (regularized-gamma-p (+ (Math/floor x) 1.0) lambda))))
  (-quantile [d p]
    (if (>= p 1.0)
      ##Inf
      (loop [k 0]
        (if (>= (-cdf d k) p) k (recur (inc k))))))
  Moments
  (-mean [_] lambda)
  (-variance [_] lambda))

(defrecord NegativeBinomial [r p]
  ;; the number of failures before the r-th success, success probability p:
  ;; a Poisson whose mean is Gamma(r, (1−p)/p)
  Distribution
  (-draw [_] (poisson-draw (* (standard-gamma r) (/ (- 1.0 p) p))))
  (-logpdf [_ k]
    (if (and (whole? k) (not (neg? k)))
      (+ (- (lgamma (+ k r)) (lgamma r) (lgamma (+ k 1.0)))
         (* r (Math/log p))
         (if (zero? k) 0.0 (* k (Math/log (- 1.0 p)))))
      ##-Inf))
  Moments
  (-mean [_] (/ (* r (- 1.0 p)) p))
  (-variance [_] (/ (* r (- 1.0 p)) (* p p))))

(defrecord Bernoulli [p]
  Distribution
  (-draw [_] (if (< (u01) p) 1 0))
  (-logpdf [_ x]
    (cond (not (number? x)) ##-Inf
          (== x 1) (Math/log p)
          (== x 0) (Math/log (- 1.0 p))
          :else ##-Inf))
  Moments
  (-mean [_] p)
  (-variance [_] (* p (- 1.0 p))))

(defrecord Flip [p]
  Distribution
  (-draw [_] (< (u01) p))
  (-logpdf [_ x]
    (case x true (Math/log p) false (Math/log (- 1.0 p)) ##-Inf)))

(defrecord Discrete [weights total]
  Distribution
  (-draw [_]
    (let [target (* (u01) total)]
      (loop [i 0 acc 0.0 [w & more] weights]
        (let [acc (+ acc w)]
          (if (or (< target acc) (empty? more)) i (recur (inc i) acc more))))))
  (-logpdf [_ i]
    (if (and (whole? i) (< -1 i (count weights)))
      (Math/log (/ (nth weights (long i)) total))
      ##-Inf))
  Univariate
  (-cdf [_ x]
    (if (neg? x) 0.0 (/ (reduce + 0.0 (take (inc (long (Math/floor x))) weights)) total)))
  (-quantile [d p]
    (loop [i 0]
      (if (or (>= (-cdf d i) p) (>= i (dec (count weights)))) i (recur (inc i))))))

(defrecord Dirichlet [alpha]
  Distribution
  (-draw [_]
    (let [gs (mapv standard-gamma alpha)
          total (reduce + gs)]
      (mapv #(/ % total) gs)))
  (-logpdf [_ x]
    (if (and (= (count x) (count alpha))
             (every? #(in? 0.0 % 1.0) x)
             (< (Math/abs (- 1.0 (reduce + x))) 1e-9))
      (- (reduce + (map (fn [a v] (xlogy (- a 1.0) v)) alpha x))
         (- (reduce + (map lgamma alpha)) (lgamma (reduce + alpha))))
      ##-Inf))
  Moments
  (-mean [_] (let [t (reduce + alpha)] (mapv #(/ % t) alpha)))
  (-variance [_]
    (let [t (reduce + alpha)]
      (mapv #(/ (* % (- t %)) (* t t (+ t 1.0))) alpha))))

(defn- finite? [x]
  (and (number? x) #?(:clj (Double/isFinite (double x)) :cljs (js/isFinite x))))

(defn- positive? [x] (and (finite? x) (pos? x)))

(defn- check!
  "Throw unless `ok?`: a distribution's parameters outside their domain would
  give meaningless draws and densities rather than an error."
  [ok? distribution parameters]
  (when-not ok?
    (throw (ex-info (str "Invalid parameters for " (name distribution))
                    {:type ::invalid-parameters
                     :distribution distribution
                     :parameters parameters}))))

(defn- weights? [ws]
  (and (seq ws) (every? #(and (finite? %) (>= % 0.0)) ws) (pos? (reduce + ws))))

(defn normal "Normal(μ, σ), σ > 0 the standard deviation." [mu sigma]
  (check! (and (finite? mu) (positive? sigma)) :normal {:mu mu :sigma sigma})
  (->Normal (double mu) (double sigma)))

(defn uniform "Uniform on [a, b], a < b." [a b]
  (check! (and (finite? a) (finite? b) (< a b)) :uniform {:a a :b b})
  (->Uniform (double a) (double b)))

(defn exponential "Exponential with rate λ > 0." [lambda]
  (check! (positive? lambda) :exponential {:lambda lambda})
  (->Exponential (double lambda)))

(defn gamma "Gamma with shape α > 0 and SCALE β > 0 (mean αβ)." [alpha beta]
  (check! (and (positive? alpha) (positive? beta)) :gamma {:alpha alpha :beta beta})
  (->Gamma (double alpha) (double beta)))

(defn beta "Beta(α, β), α, β > 0." [alpha beta]
  (check! (and (positive? alpha) (positive? beta)) :beta {:alpha alpha :beta beta})
  (->Beta (double alpha) (double beta)))

(defn negative-binomial
  "Failures before the r-th success, success probability p ∈ (0, 1]; r > 0
  need not be whole."
  [r p]
  (check! (and (positive? r) (finite? p) (< 0.0 p) (<= p 1.0)) :negative-binomial {:r r :p p})
  (->NegativeBinomial (double r) (double p)))

(defn poisson "Poisson with mean λ > 0." [lambda]
  (check! (positive? lambda) :poisson {:lambda lambda})
  (->Poisson (double lambda)))

(defn bernoulli "1 with probability p, else 0." [p]
  (check! (and (finite? p) (<= 0.0 p 1.0)) :bernoulli {:p p})
  (->Bernoulli (double p)))

(defn flip "true with probability p." [p]
  (check! (and (finite? p) (<= 0.0 p 1.0)) :flip {:p p})
  (->Flip (double p)))

(defn discrete "An index i with probability weights[i] / Σ weights; weights ≥ 0." [weights]
  (check! (weights? weights) :discrete {:weights weights})
  (let [ws (mapv double weights)] (->Discrete ws (reduce + ws))))

(defn dirichlet "Dirichlet(α) over the simplex, every αᵢ > 0." [alpha]
  (check! (and (seq alpha) (every? positive? alpha)) :dirichlet {:alpha alpha})
  (->Dirichlet (mapv double alpha)))

;; =============================================================================
;; More distributions
;; =============================================================================

(defrecord Categorical [values weights total]
  Distribution
  (-draw [_]
    (let [target (* (u01) total)]
      (loop [i 0 acc 0.0]
        (let [acc (+ acc (nth weights i))]
          (if (or (< target acc) (= i (dec (count weights))))
            (nth values i)
            (recur (inc i) acc))))))
  (-logpdf [_ x]
    (if-let [i (first (keep-indexed (fn [i v] (when (= v x) i)) values))]
      (Math/log (/ (nth weights i) total))
      ##-Inf)))

(defrecord StudentT [nu mu sigma]
  Distribution
  (-draw [_]
    (let [chi2 (* 2.0 (standard-gamma (* 0.5 nu)))]
      (+ mu (* sigma (/ (standard-normal) (Math/sqrt (/ chi2 nu)))))))
  (-logpdf [_ x]
    (let [t (/ (- x mu) sigma)]
      (- (lgamma (* 0.5 (+ nu 1.0)))
         (lgamma (* 0.5 nu))
         (* 0.5 (Math/log (* nu Math/PI)))
         (Math/log sigma)
         (* 0.5 (+ nu 1.0) (Math/log (+ 1.0 (/ (* t t) nu)))))))
  Moments
  (-mean [_] (if (> nu 1.0) mu ##NaN))
  (-variance [_] (cond (> nu 2.0) (/ (* sigma sigma nu) (- nu 2.0))
                       (> nu 1.0) ##Inf
                       :else ##NaN)))

(defrecord ChiSquared [k]
  Distribution
  (-draw [_] (* 2.0 (standard-gamma (* 0.5 k))))
  (-logpdf [_ x]
    (if (<= x 0.0)
      ##-Inf
      (- (* (- (* 0.5 k) 1.0) (Math/log x)) (* 0.5 x)
         (lgamma (* 0.5 k)) (* 0.5 k (Math/log 2.0)))))
  Univariate
  (-cdf [_ x] (regularized-gamma-p (* 0.5 k) (* 0.5 x)))
  (-quantile [_ _]
    (throw (ex-info "ChiSquared has no quantile yet" {:type ::unsupported})))
  Moments
  (-mean [_] k)
  (-variance [_] (* 2.0 k)))

(defn- cholesky
  "The lower-triangular L with L Lᵀ = `a` (a symmetric positive-definite
  matrix as vectors of rows)."
  [a]
  (let [n (count a)]
    (reduce
     (fn [l [i j]]
       (let [s (reduce + 0.0 (map #(* (get-in l [i %]) (get-in l [j %])) (range j)))
             v (if (= i j)
                 (let [d (- (get-in a [i i]) s)]
                   (when-not (pos? d)
                     (throw (ex-info "The covariance is not positive definite"
                                     {:type ::not-positive-definite})))
                   (Math/sqrt d))
                 (/ (- (get-in a [i j]) s) (get-in l [j j])))]
         (assoc-in l [i j] v)))
     (vec (repeat n (vec (repeat n 0.0))))
     (for [i (range n) j (range (inc i))] [i j]))))

(defrecord MultivariateNormal [mean cov chol]
  Distribution
  (-draw [_]
    (let [z (vec (repeatedly (count mean) standard-normal))]
      (mapv (fn [m row] (+ m (reduce + (map * row z)))) mean chol)))
  (-logpdf [_ x]
    ;; solve L y = x − μ by forward substitution: log p = −½(d log 2π + 2 Σ log Lᵢᵢ + |y|²)
    (let [n (count mean)
          r (mapv - x mean)
          y (reduce (fn [y i]
                      (conj y (/ (- (nth r i) (reduce + 0.0 (map #(* (get-in chol [i %]) (nth y %)) (range i))))
                                 (get-in chol [i i]))))
                    [] (range n))]
      (* -0.5 (+ (* n log-2pi)
                 (* 2.0 (reduce + (map #(Math/log (get-in chol [% %])) (range n))))
                 (reduce + (map #(* % %) y))))))
  Moments
  (-mean [_] mean)
  (-variance [_] (mapv #(get-in cov [% %]) (range (count mean)))))

(defn categorical
  "A value with probability ∝ its weight: `outcomes` a map {value weight}
  or a sequence of [value weight] pairs."
  [outcomes]
  (let [pairs (vec (seq outcomes))
        _ (check! (weights? (map second pairs)) :categorical {:outcomes outcomes})
        ;; an outcome listed twice carries both weights
        values (vec (distinct (map first pairs)))
        by-value (reduce (fn [m [v w]] (update m v (fnil + 0.0) (double w))) {} pairs)
        ws (mapv by-value values)]
    (->Categorical values ws (reduce + ws))))

(defn student-t
  "Student's t with ν degrees of freedom, location μ and scale σ."
  ([nu] (student-t nu 0.0 1.0))
  ([nu mu sigma]
   (check! (and (positive? nu) (finite? mu) (positive? sigma)) :student-t
           {:nu nu :mu mu :sigma sigma})
   (->StudentT (double nu) (double mu) (double sigma))))

(defn chi-squared "χ² with k > 0 degrees of freedom." [k]
  (check! (positive? k) :chi-squared {:k k})
  (->ChiSquared (double k)))

(defn mvn
  "Multivariate normal with `mean` (a vector) and covariance `cov` (vectors of
  rows, symmetric positive definite)."
  [mean cov]
  (check! (and (seq mean) (every? finite? mean)
               (= (count mean) (count cov))
               (every? #(= (count mean) (count %)) cov))
          :mvn {:mean mean :cov cov})
  (let [cov (mapv #(mapv double %) cov)]
    (->MultivariateNormal (mapv double mean) cov (cholesky cov))))

;; =============================================================================
;; Support
;; =============================================================================

(defprotocol Continuous
  (-continuous? [d] "Whether `d` is a law on (an interval of) the reals."))

(extend-protocol Continuous
  #?(:clj Object :cljs default)
  (-continuous? [_] false)
  Normal (-continuous? [_] true)
  Uniform (-continuous? [_] true)
  Exponential (-continuous? [_] true)
  Gamma (-continuous? [_] true)
  Beta (-continuous? [_] true)
  StudentT (-continuous? [_] true)
  ChiSquared (-continuous? [_] true))

(defn continuous?
  "Whether `d` is a law on (an interval of) the reals: a scalar site a
  random walk can move."
  [d]
  (-continuous? d))
