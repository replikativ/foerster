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

;; =============================================================================
;; Binomial, LogNormal, HalfNormal, Cauchy, HalfCauchy, UniformDiscrete
;; =============================================================================

(defn- lattice-cdf
  "P(X ≤ x) of a law on 0..n, by summing its mass."
  [d n x]
  (cond (neg? x) 0.0
        (>= x n) 1.0
        :else (min 1.0 (reduce + 0.0 (map #(Math/exp (-logpdf d %)) (range (inc (long (Math/floor x)))))))))

(defn- lattice-quantile [d n p]
  (loop [k 0 acc 0.0]
    (let [acc (+ acc (Math/exp (-logpdf d k)))]
      (if (or (>= acc p) (>= k n)) k (recur (inc k) acc)))))

(defrecord Binomial [n p]
  ;; successes in n trials of success probability p
  Distribution
  (-draw [_] (loop [i 0 k 0] (if (= i n) k (recur (inc i) (if (< (u01) p) (inc k) k)))))
  (-logpdf [_ k]
    (if (and (whole? k) (<= 0 k n))
      (+ (- (lgamma (+ n 1.0)) (lgamma (+ k 1.0)) (lgamma (+ (- n k) 1.0)))
         (xlogy k p) (xlogy (- n k) (- 1.0 p)))
      ##-Inf))
  Univariate
  (-cdf [d x] (lattice-cdf d n x))
  (-quantile [d p] (lattice-quantile d n p))
  Moments
  (-mean [_] (* n p))
  (-variance [_] (* n p (- 1.0 p))))

(defrecord LogNormal [mu sigma]
  ;; exp of a Normal(μ, σ)
  Distribution
  (-draw [_] (Math/exp (+ mu (* sigma (standard-normal)))))
  (-logpdf [_ x]
    (if (pos? x)
      (let [z (/ (- (Math/log x) mu) sigma)]
        (- (* -0.5 z z) (Math/log sigma) log-sqrt-2pi (Math/log x)))
      ##-Inf))
  Univariate
  (-cdf [_ x] (if (pos? x) (normal-cdf (/ (- (Math/log x) mu) sigma)) 0.0))
  (-quantile [_ p] (Math/exp (+ mu (* sigma (normal-quantile p)))))
  Moments
  (-mean [_] (Math/exp (+ mu (* 0.5 sigma sigma))))
  (-variance [_] (* (- (Math/exp (* sigma sigma)) 1.0) (Math/exp (+ (* 2.0 mu) (* sigma sigma))))))

(defrecord HalfNormal [sigma]
  ;; |Normal(0, σ)|
  Distribution
  (-draw [_] (Math/abs (* sigma (standard-normal))))
  (-logpdf [_ x]
    (if (neg? x)
      ##-Inf
      (let [z (/ x sigma)]
        (- (Math/log 2.0) (* 0.5 z z) (Math/log sigma) log-sqrt-2pi))))
  Univariate
  (-cdf [_ x] (if (neg? x) 0.0 (- (* 2.0 (normal-cdf (/ x sigma))) 1.0)))
  (-quantile [_ p] (* sigma (normal-quantile (* 0.5 (+ 1.0 p)))))
  Moments
  (-mean [_] (* sigma (Math/sqrt (/ 2.0 Math/PI))))
  (-variance [_] (* sigma sigma (- 1.0 (/ 2.0 Math/PI)))))

(defrecord Cauchy [location scale]
  Distribution
  (-draw [_] (+ location (* scale (Math/tan (* Math/PI (- (u01) 0.5))))))
  (-logpdf [_ x]
    (let [z (/ (- x location) scale)]
      (- (+ (Math/log (* Math/PI scale)) (Math/log (+ 1.0 (* z z)))))))
  Univariate
  (-cdf [_ x] (+ 0.5 (/ (Math/atan (/ (- x location) scale)) Math/PI)))
  (-quantile [_ p] (+ location (* scale (Math/tan (* Math/PI (- p 0.5)))))))

(defrecord HalfCauchy [scale]
  ;; |Cauchy(0, γ)|
  Distribution
  (-draw [_] (Math/abs (* scale (Math/tan (* Math/PI (- (u01) 0.5))))))
  (-logpdf [_ x]
    (if (neg? x)
      ##-Inf
      (let [z (/ x scale)]
        (- (Math/log 2.0) (Math/log (* Math/PI scale)) (Math/log (+ 1.0 (* z z)))))))
  Univariate
  (-cdf [_ x] (if (neg? x) 0.0 (/ (* 2.0 (Math/atan (/ x scale))) Math/PI)))
  (-quantile [_ p] (* scale (Math/tan (* 0.5 Math/PI p)))))

(defrecord UniformDiscrete [a b]
  ;; the integers a, a+1, …, b−1, each with probability 1/(b−a)
  Distribution
  (-draw [_] (+ a (long (Math/floor (* (u01) (- b a))))))
  (-logpdf [_ k]
    (if (and (whole? k) (<= a k) (< k b)) (- (Math/log (- b a))) ##-Inf))
  Univariate
  (-cdf [_ x] (cond (< x a) 0.0 (>= x b) 1.0 :else (/ (- (+ (Math/floor x) 1.0) a) (- b a))))
  (-quantile [_ p] (min (dec b) (+ a (long (Math/floor (* p (- b a)))))))
  Moments
  (-mean [_] (* 0.5 (+ a b -1)))
  (-variance [_] (/ (- (* (- b a) (- b a)) 1.0) 12.0)))

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

(defn binomial "Successes in n ≥ 0 trials of success probability p ∈ [0, 1]." [n p]
  (check! (and (whole? n) (>= n 0) (finite? p) (<= 0.0 p 1.0)) :binomial {:n n :p p})
  (->Binomial (long n) (double p)))

(defn log-normal "exp of Normal(μ, σ), σ > 0." [mu sigma]
  (check! (and (finite? mu) (positive? sigma)) :log-normal {:mu mu :sigma sigma})
  (->LogNormal (double mu) (double sigma)))

(defn half-normal "|Normal(0, σ)|, σ > 0: a prior for a scale." [sigma]
  (check! (positive? sigma) :half-normal {:sigma sigma})
  (->HalfNormal (double sigma)))

(defn cauchy "Cauchy with location x₀ and scale γ > 0." [location scale]
  (check! (and (finite? location) (positive? scale)) :cauchy {:location location :scale scale})
  (->Cauchy (double location) (double scale)))

(defn half-cauchy "|Cauchy(0, γ)|, γ > 0: a heavy-tailed prior for a scale." [scale]
  (check! (positive? scale) :half-cauchy {:scale scale})
  (->HalfCauchy (double scale)))

(defn uniform-discrete "The integers a, a+1, …, b−1, equally likely (a < b)." [a b]
  (check! (and (whole? a) (whole? b) (< a b)) :uniform-discrete {:a a :b b})
  (->UniformDiscrete (long a) (long b)))

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

(defprotocol Finite
  (-support [d] "The values of a distribution with finite support, in order;
  nil when its support is infinite or continuous."))

(extend-protocol Finite
  #?(:clj Object :cljs default)
  (-support [_] nil)
  Bernoulli (-support [_] [0 1])
  Flip (-support [_] [true false])
  Discrete (-support [d] (vec (range (count (:weights d)))))
  Categorical (-support [d] (:values d))
  Binomial (-support [d] (vec (range (inc (:n d)))))
  UniformDiscrete (-support [d] (vec (range (:a d) (:b d)))))

(defn support
  "The values `d` can take, when finitely many (exact enumeration walks
  them); nil otherwise."
  [d]
  (-support d))

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
  ChiSquared (-continuous? [_] true)
  LogNormal (-continuous? [_] true)
  HalfNormal (-continuous? [_] true)
  Cauchy (-continuous? [_] true)
  HalfCauchy (-continuous? [_] true))

(defn continuous?
  "Whether `d` is a law on (an interval of) the reals: a scalar site a
  random walk can move."
  [d]
  (-continuous? d))

;; =============================================================================
;; Weibull, Laplace, inverse gamma, Gumbel, half-t, beta-binomial
;; =============================================================================

(def ^:private euler-gamma 0.5772156649015329)

(defrecord Weibull [k lambda]
  ;; shape k, scale λ
  Distribution
  (-draw [_] (* lambda (Math/pow (- (Math/log (- 1.0 (u01)))) (/ 1.0 k))))
  (-logpdf [_ x]
    (if (neg? x)
      ##-Inf
      (let [z (/ x lambda)]
        (- (+ (Math/log (/ k lambda)) (xlogy (- k 1.0) z)) (Math/pow z k)))))
  Univariate
  (-cdf [_ x] (if (neg? x) 0.0 (- 1.0 (Math/exp (- (Math/pow (/ x lambda) k))))))
  (-quantile [_ p] (* lambda (Math/pow (- (Math/log (- 1.0 p))) (/ 1.0 k))))
  Moments
  (-mean [_] (* lambda (Math/exp (lgamma (+ 1.0 (/ 1.0 k))))))
  (-variance [_] (* lambda lambda (- (Math/exp (lgamma (+ 1.0 (/ 2.0 k))))
                                     (Math/exp (* 2.0 (lgamma (+ 1.0 (/ 1.0 k)))))))))

(defrecord Laplace [mu b]
  Distribution
  (-draw [d] (-quantile d (u01)))
  (-logpdf [_ x] (- (- (Math/log (* 2.0 b))) (/ (Math/abs (- x mu)) b)))
  Univariate
  (-cdf [_ x] (if (< x mu) (* 0.5 (Math/exp (/ (- x mu) b))) (- 1.0 (* 0.5 (Math/exp (/ (- mu x) b))))))
  (-quantile [_ p] (if (< p 0.5) (+ mu (* b (Math/log (* 2.0 p)))) (- mu (* b (Math/log (* 2.0 (- 1.0 p)))))))
  Moments
  (-mean [_] mu)
  (-variance [_] (* 2.0 b b)))

(defrecord InverseGamma [alpha beta]
  ;; 1/X for X ~ Gamma(α, rate β): shape α, scale β
  Distribution
  (-draw [_] (/ beta (standard-gamma alpha)))
  (-logpdf [_ x]
    (if (<= x 0.0)
      ##-Inf
      (- (* alpha (Math/log beta)) (lgamma alpha) (* (+ alpha 1.0) (Math/log x)) (/ beta x))))
  Univariate
  (-cdf [_ x] (if (<= x 0.0) 0.0 (regularized-gamma-q alpha (/ beta x))))
  (-quantile [_ _]
    (throw (ex-info "InverseGamma has no quantile yet" {:type ::unsupported})))
  Moments
  (-mean [_] (if (> alpha 1.0) (/ beta (- alpha 1.0)) ##Inf))
  (-variance [_] (if (> alpha 2.0)
                   (/ (* beta beta) (* (- alpha 1.0) (- alpha 1.0) (- alpha 2.0)))
                   ##Inf)))

(defrecord Gumbel [mu beta]
  ;; the maximum's law: F(x) = exp(−exp(−(x − μ)/β))
  Distribution
  (-draw [d] (-quantile d (u01)))
  (-logpdf [_ x] (let [z (/ (- x mu) beta)] (- (- (Math/log beta)) z (Math/exp (- z)))))
  Univariate
  (-cdf [_ x] (Math/exp (- (Math/exp (- (/ (- x mu) beta))))))
  (-quantile [_ p] (- mu (* beta (Math/log (- (Math/log p))))))
  Moments
  (-mean [_] (+ mu (* beta euler-gamma)))
  (-variance [_] (/ (* Math/PI Math/PI beta beta) 6.0)))

(defrecord HalfStudentT [nu sigma]
  ;; |Student-t(ν, 0, σ)|
  Distribution
  (-draw [_]
    (let [chi2 (* 2.0 (standard-gamma (* 0.5 nu)))]
      (Math/abs (* sigma (/ (standard-normal) (Math/sqrt (/ chi2 nu)))))))
  (-logpdf [_ x]
    (if (neg? x)
      ##-Inf
      (let [t (/ x sigma)]
        (- (+ (Math/log 2.0) (lgamma (* 0.5 (+ nu 1.0))))
           (lgamma (* 0.5 nu))
           (* 0.5 (Math/log (* nu Math/PI)))
           (Math/log sigma)
           (* 0.5 (+ nu 1.0) (Math/log (+ 1.0 (/ (* t t) nu))))))))
  Moments
  (-mean [_] (if (> nu 1.0)
               (* 2.0 sigma (Math/sqrt (/ nu Math/PI))
                  (Math/exp (- (lgamma (* 0.5 (+ nu 1.0))) (lgamma (* 0.5 nu)) (Math/log (- nu 1.0)))))
               ##Inf))
  (-variance [d] (if (> nu 2.0)
                   (- (/ (* sigma sigma nu) (- nu 2.0)) (let [m (-mean d)] (* m m)))
                   ##Inf)))

(defrecord BetaBinomial [n alpha beta]
  ;; successes in n trials whose probability is Beta(α, β)
  Distribution
  (-draw [_]
    (let [x (standard-gamma alpha) y (standard-gamma beta) p (/ x (+ x y))]
      (loop [i 0 k 0] (if (= i n) k (recur (inc i) (if (< (u01) p) (inc k) k))))))
  (-logpdf [_ k]
    (if (and (whole? k) (<= 0 k n))
      (+ (- (lgamma (+ n 1.0)) (lgamma (+ k 1.0)) (lgamma (+ (- n k) 1.0)))
         (- (lbeta (+ k alpha) (+ (- n k) beta)) (lbeta alpha beta)))
      ##-Inf))
  Univariate
  (-cdf [d x] (lattice-cdf d n x))
  (-quantile [d p] (lattice-quantile d n p))
  Moments
  (-mean [_] (/ (* n alpha) (+ alpha beta)))
  (-variance [_] (let [s (+ alpha beta)]
                   (/ (* n alpha beta (+ s n)) (* s s (+ s 1.0))))))

;; =============================================================================
;; Truncated and censored laws, zero inflation and hurdles
;; =============================================================================

(defn- below
  "P(X < x): F(x) for a continuous law, F(x − 1) on the integers."
  [d x]
  (cond (= x ##-Inf) 0.0
        (= x ##Inf) 1.0
        (-continuous? d) (-cdf d x)
        :else (-cdf d (dec (Math/ceil x)))))

(defn- upto
  "P(X ≤ x)."
  [d x]
  (cond (= x ##-Inf) 0.0 (= x ##Inf) 1.0 :else (-cdf d x)))

(defrecord Truncated [d lo hi below-lo log-mass]
  ;; d restricted to [lo, hi], renormalized
  Distribution
  (-draw [this]
    (if (or (> log-mass (Math/log 0.05)) (not (-continuous? d)))
      (loop [] (let [x (-draw d)] (if (<= lo x hi) x (recur))))
      (-quantile this (u01))))
  (-logpdf [_ x] (if (<= lo x hi) (- (-logpdf d x) log-mass) ##-Inf))
  Univariate
  (-cdf [_ x] (cond (< x lo) 0.0
                    (>= x hi) 1.0
                    :else (/ (- (-cdf d x) below-lo) (Math/exp log-mass))))
  (-quantile [_ p] (min hi (max lo (-quantile d (+ below-lo (* p (Math/exp log-mass))))))))

(defrecord Censored [d lo hi]
  ;; d observed through a clamp to [lo, hi]: a value at a bound carries the
  ;; mass beyond it
  Distribution
  (-draw [_] (min hi (max lo (-draw d))))
  (-logpdf [_ x]
    (cond (or (< x lo) (> x hi)) ##-Inf
          (== x lo) (Math/log (upto d lo))
          (== x hi) (Math/log1p (- (below d hi)))
          :else (-logpdf d x))))

(defrecord ZeroInflated [p-zero d]
  ;; 0 with probability p-zero, otherwise a draw of d (which may be 0 too,
  ;; when d is discrete); for a continuous d, a point mass at 0 next to a
  ;; density
  Distribution
  (-draw [_] (if (< (u01) p-zero) 0 (-draw d)))
  (-logpdf [_ x]
    (let [inner (+ (Math/log1p (- p-zero)) (-logpdf d x))]
      (if (and (number? x) (zero? x))
        (if (-continuous? d)
          (Math/log p-zero)
          (let [a (Math/log p-zero) hi (max a inner)]
            (if (= hi ##-Inf) ##-Inf (+ hi (Math/log (+ (Math/exp (- a hi)) (Math/exp (- inner hi))))))))
        inner)))
  Moments
  (-mean [_] (* (- 1.0 p-zero) (-mean d)))
  (-variance [_] (let [m (-mean d)]
                   (+ (* (- 1.0 p-zero) (-variance d)) (* p-zero (- 1.0 p-zero) m m)))))

(defrecord Hurdle [p-zero d log-positive]
  ;; 0 with probability p-zero, otherwise d conditioned on being nonzero
  Distribution
  (-draw [_] (if (< (u01) p-zero) 0 (loop [] (let [x (-draw d)] (if (zero? x) (recur) x)))))
  (-logpdf [_ x]
    (if (and (number? x) (zero? x))
      (Math/log p-zero)
      (- (+ (Math/log1p (- p-zero)) (-logpdf d x)) log-positive))))

(defrecord ZeroSumNormal [sigma n]
  ;; n normals of scale σ constrained to sum to zero: the density on that
  ;; (n−1)-dimensional subspace, as PyMC's ZeroSumNormal
  Distribution
  (-draw [_]
    (let [z (vec (repeatedly n #(* sigma (standard-normal))))
          m (/ (reduce + z) n)]
      (mapv #(- % m) z)))
  (-logpdf [_ x]
    (if (and (= n (count x)) (< (Math/abs (reduce + x)) (* 1e-9 n (max 1.0 (reduce max (map #(Math/abs (double %)) x))))))
      (- (* -0.5 (reduce + (map #(let [z (/ % sigma)] (* z z)) x)))
         (* (dec n) (+ log-sqrt-2pi (Math/log sigma))))
      ##-Inf)))

(defn weibull "Weibull with shape k > 0 and scale λ > 0." [k lambda]
  (check! (and (positive? k) (positive? lambda)) :weibull {:k k :lambda lambda})
  (->Weibull (double k) (double lambda)))

(defn laplace "Laplace (double exponential) with location μ and scale b > 0." [mu b]
  (check! (and (finite? mu) (positive? b)) :laplace {:mu mu :b b})
  (->Laplace (double mu) (double b)))

(defn inverse-gamma "1/X for X ~ Gamma: shape α > 0 and scale β > 0 (mean β/(α − 1))." [alpha beta]
  (check! (and (positive? alpha) (positive? beta)) :inverse-gamma {:alpha alpha :beta beta})
  (->InverseGamma (double alpha) (double beta)))

(defn gumbel "Gumbel (extreme value) with location μ and scale β > 0." [mu beta]
  (check! (and (finite? mu) (positive? beta)) :gumbel {:mu mu :beta beta})
  (->Gumbel (double mu) (double beta)))

(defn half-student-t "|Student-t(ν, 0, σ)|: a prior for a scale, between half-normal and half-Cauchy." [nu sigma]
  (check! (and (positive? nu) (positive? sigma)) :half-student-t {:nu nu :sigma sigma})
  (->HalfStudentT (double nu) (double sigma)))

(defn beta-binomial "Successes in n ≥ 0 trials of a Beta(α, β) probability: an overdispersed binomial." [n alpha beta]
  (check! (and (whole? n) (>= n 0) (positive? alpha) (positive? beta)) :beta-binomial {:n n :alpha alpha :beta beta})
  (->BetaBinomial (long n) (double alpha) (double beta)))

(defn gamma-mean-sd
  "The gamma with mean m > 0 and standard deviation sd > 0 (shape (m/sd)²,
  scale sd²/m)."
  [m sd]
  (check! (and (positive? m) (positive? sd)) :gamma-mean-sd {:mean m :sd sd})
  (gamma (/ (* m m) (* sd sd)) (/ (* sd sd) m)))

(defn truncated
  "`d` (a univariate law with a `cdf`) restricted to [lo, hi] and
  renormalized; lo or hi may be ##-Inf / ##Inf. A discrete `d` lives on the
  integers."
  [d lo hi]
  (check! (and (satisfies? Univariate d) (< lo hi)) :truncated {:d d :lo lo :hi hi})
  (let [below-lo (below d lo)
        log-mass (Math/log (- (upto d hi) below-lo))]
    (check! (> log-mass ##-Inf) :truncated {:d d :lo lo :hi hi :mass 0.0})
    (->Truncated d (double lo) (double hi) below-lo log-mass)))

(defn censored
  "`d` (a univariate law with a `cdf`) observed through a clamp to [lo, hi]:
  an observation at a bound has the probability of lying beyond it — the
  likelihood of data that saturate a detection limit."
  [d lo hi]
  (check! (and (satisfies? Univariate d) (< lo hi)) :censored {:d d :lo lo :hi hi})
  (->Censored d (double lo) (double hi)))

(defn zero-inflated
  "0 with probability `p-zero`, otherwise a draw of `d`: extra zeros on top
  of d's own (a zero-inflated Poisson or negative binomial)."
  [p-zero d]
  (check! (and (finite? p-zero) (<= 0.0 p-zero 1.0) (distribution? d)) :zero-inflated {:p-zero p-zero})
  (->ZeroInflated (double p-zero) d))

(defn hurdle
  "0 with probability `p-zero`, otherwise `d` conditioned to be nonzero:
  whether anything happens, and how much if it does, as separate parts."
  [p-zero d]
  (check! (and (finite? p-zero) (<= 0.0 p-zero 1.0) (distribution? d)) :hurdle {:p-zero p-zero})
  (let [p0 (if (-continuous? d) 0.0 (Math/exp (-logpdf d 0)))]
    (check! (< p0 1.0) :hurdle {:p-zero p-zero :d d})
    (->Hurdle (double p-zero) d (if (zero? p0) 0.0 (Math/log1p (- p0))))))

(defn zero-sum-normal
  "n ≥ 2 normals of scale σ constrained to sum to zero (as PyMC's
  ZeroSumNormal): identifiable group offsets next to an intercept."
  [sigma n]
  (check! (and (positive? sigma) (whole? n) (>= n 2)) :zero-sum-normal {:sigma sigma :n n})
  (->ZeroSumNormal (double sigma) (long n)))

;; the new laws' support and continuity

(extend-protocol Finite
  BetaBinomial (-support [d] (vec (range (inc (:n d)))))
  Truncated (-support [{:keys [d lo hi]}] (some->> (-support d) (filterv #(<= lo % hi))))
  ZeroInflated (-support [{:keys [d]}] (some->> (-support d) (cons 0) distinct vec))
  Hurdle (-support [{:keys [d]}] (some->> (-support d) (cons 0) distinct vec)))

(extend-protocol Continuous
  Weibull (-continuous? [_] true)
  Laplace (-continuous? [_] true)
  InverseGamma (-continuous? [_] true)
  Gumbel (-continuous? [_] true)
  HalfStudentT (-continuous? [_] true)
  Truncated (-continuous? [{:keys [d]}] (-continuous? d)))
