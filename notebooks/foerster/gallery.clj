;; # A Gallery of Classic Models

;; Seven models from the classic collections of
;; [Anglican](https://github.com/probprog/anglican) and
;; [Stan](https://mc-stan.org/docs/stan-users-guide/), each run through
;; `infer/infer` and set next to an answer computed without sampling: a
;; closed form, an enumeration or a quadrature. Each section also shows one
;; diagnostic — R-hat and effective sample size for Markov chains, the
;; evidence or the surviving histories for SMC, or a posterior predictive
;; check — and what it says when a method goes wrong. Every number below is
;; computed as the notebook renders, with fixed seeds.

(ns foerster.gallery
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as diagnostics]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe factor]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

(def world (sp/create-execution-context))

(defn run
  "Run the inference `(make)` returns, seeded, and time it."
  [seed make]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        measure (sp/with-context world @(make))]
    {:measure measure
     :ms (long (/ (- (System/nanoTime) t0) 1e6))}))

(defn r3
  "`x` rounded to three decimals, for tables."
  [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

(defn distribution
  "The weighted distribution of `(f value)` over the particles: {x p}."
  [measure f]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))]
    (reduce (fn [acc [w [particle _]]] (update acc (f (m/get-value particle)) (fnil + 0.0) w))
            {} (map vector ws ps))))

(defn total-variation
  "Total variation distance between two distributions {x p}."
  [p q]
  (* 0.5 (reduce + (map #(Math/abs (- (get p % 0.0) (get q % 0.0)))
                        (into (set (keys p)) (keys q))))))

;; ## 1. The pencil factory
;;
;; A factory makes pencils, a fraction p of them defective; an inspector
;; tests ten and finds three defective. With a uniform prior on p, the
;; posterior is Beta(1 + 3, 1 + 7), and the evidence p(3 of 10) is 1/11 —
;; under a uniform prior every count from 0 to 10 is equally likely.

(defn pencils []
  (spin
   (let [p (sample (dist/beta 1.0 1.0) :id :p)]
     (observe (dist/binomial 10 p) 3 :id :defective)
     p)))

(def pencil-runs
  (array-map
   "importance sampling" (run 1 #(infer/infer (pencils) {:method :importance :particles 2000}))
   "SMC" (run 1 #(infer/infer (pencils) {:method :smc :particles 2000}))))

(kind/table
 {:column-names ["" "mean" "sd" "log evidence" "ESS" "ms"]
  :row-vectors (into [["exact Beta(4, 8)" (r3 (/ 4.0 12)) (r3 (Math/sqrt (/ (* 4.0 8) (* 12 12 13))))
                       (r3 (Math/log (/ 1.0 11))) "–" "–"]]
                     (for [[label {:keys [measure ms]}] pencil-runs]
                       (let [{:keys [mean sd ess]} (diagnostics/summary measure identity)]
                         [label (r3 mean) (r3 sd) (r3 (m/log-marginal measure)) (r3 ess) ms])))})

;; The diagnostic here is the evidence: both estimate log(1/11). The
;; weight-based ESS means something for importance sampling — how many of
;; the 2000 prior draws matter. SMC has resampled at the observation, so its
;; ESS counts particles, not distinct draws.

;; ## 2. Bayesian linear regression
;;
;; Ten points around a line, an unknown intercept and slope with N(0, 10²)
;; priors, and noise of known standard deviation 1. With the noise known,
;; the posterior is Gaussian and exact. The inputs are centred, which makes
;; intercept and slope nearly independent a posteriori.

(def xs [-4.5 -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5])
(def ys [1.2 2.6 5.3 6.9 9.4 10.6 13.4 15.1 16.8 19.5])

(defn regression []
  (spin
   (let [a (sample (dist/normal 0.0 10.0) :id :intercept)
         b (sample (dist/normal 0.0 10.0) :id :slope)]
     (loop [i 0]
       (when (< i (count xs))
         (observe (dist/normal (+ a (* b (nth xs i))) 1.0) (nth ys i) :id [:y i])
         (recur (inc i))))
     {:intercept a :slope b})))

;; The exact posterior: precision Λ = XᵀX + I/100 (noise variance 1), mean
;; Λ⁻¹Xᵀy, for X with a column of ones and the inputs.

(def regression-exact
  (let [n (count xs) sx (reduce + xs) sxx (reduce + (map * xs xs))
        sy (reduce + ys) sxy (reduce + (map * xs ys))
        l11 (+ n 0.01) l12 sx l22 (+ sxx 0.01)
        det (- (* l11 l22) (* l12 l12))
        c11 (/ l22 det) c12 (/ (- l12) det) c22 (/ l11 det)]
    {:intercept {:mean (+ (* c11 sy) (* c12 sxy)) :sd (Math/sqrt c11)}
     :slope {:mean (+ (* c12 sy) (* c22 sxy)) :sd (Math/sqrt c22)}}))

;; Four random-walk Metropolis–Hastings chains, each starting from a prior
;; draw, 2000 moves each, the first 500 discarded:

(def regression-run
  (run 2 #(infer/infer (regression) {:method :rmh :iterations 2000 :burn 500 :chains 4 :step-size 0.4})))

(kind/table
 {:column-names ["" "exact mean" "exact sd" "mean" "sd" "R-hat" "bulk ESS" "tail ESS"]
  :row-vectors (for [k [:intercept :slope]]
                 (let [s (diagnostics/summary (:measure regression-run) k)]
                   [(name k) (r3 (get-in regression-exact [k :mean])) (r3 (get-in regression-exact [k :sd]))
                    (r3 (:mean s)) (r3 (:sd s)) (r3 (:rhat s)) (r3 (:ess-bulk s)) (r3 (:ess-tail s))]))})

;; R-hat near 1 and a few hundred effective draws: the chains agree, and the
;; estimates sit within their Monte Carlo error of the exact values.
;;
;; **A posterior predictive check.** `infer/predictive` replays posterior
;; draws with the latents held and every observation drawing a fresh value.
;; If the model fits, each data point should look like a typical replicate:
;; its rank among the replicates — the fraction below it — should not sit
;; at 0 or 1.

(def replicates
  (do (random/set-seed! 3)
      (sp/with-context world @(infer/predictive (regression) (:measure regression-run) 400))))

(kind/table
 {:column-names ["x" "y" "replicates: 5%" "50%" "95%" "fraction below y"]
  :row-vectors (for [i (range (count xs))]
                 (let [reps (sort (map #(get-in % [:observations [:y i]]) replicates))
                       q (fn [p] (nth reps (int (* p (dec (count reps))))))]
                   [(nth xs i) (nth ys i) (r3 (q 0.05)) (r3 (q 0.5)) (r3 (q 0.95))
                    (r3 (/ (count (filter #(< % (nth ys i)) reps)) (double (count reps))))]))})

;; ## 3. Seven scientists
;;
;; Seven scientists measure the same quantity with instruments of very
;; different quality: −27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056
;; (MacKay, *Information Theory, Inference, and Learning Algorithms*). What is the quantity, and how noisy is each
;; scientist? Every scientist gets their own precision, drawn from a
;; common Gamma(1, 1) prior.

(def measurements [-27.020 3.570 8.191 9.898 9.603 9.945 10.056])

(defn scientists []
  (spin
   (let [mu (sample (dist/normal 0.0 50.0) :id :mu)]
     (loop [i 0 sigmas []]
       (if (< i 7)
         (let [precision (sample (dist/gamma 1.0 1.0) :id [:precision i])
               sigma (/ 1.0 (Math/sqrt precision))]
           (observe (dist/normal mu sigma) (nth measurements i) :id [:y i])
           (recur (inc i) (conj sigmas sigma)))
         {:mu mu :sigmas sigmas})))))

;; Integrating out a Gamma(1, rate 1) precision turns each measurement's
;; likelihood into a Student-t with 2 degrees of freedom, so the posterior of
;; μ is one-dimensional and a fine grid computes it. Given μ, a scientist's
;; precision is Gamma(3/2, rate 1 + (yᵢ − μ)²/2), whose E[1/√precision] is
;; closed-form; averaging it over μ's grid gives each scientist's posterior
;; mean σ. The evidence is the grid's integral.

(def scientists-exact
  (let [h 0.001
        mus (mapv #(+ -20.0 (* h %)) (range 50001))
        log-post (fn [mu] (+ (dist/logpdf (dist/normal 0.0 50.0) mu)
                             (reduce + (map #(dist/logpdf (dist/student-t 2.0 mu 1.0) %) measurements))))
        lps (mapv log-post mus)
        z (m/log-sum-exp lps)
        ws (mapv #(Math/exp (- % z)) lps)
        e (fn [g] (reduce + (map (fn [w mu] (* w (g mu))) ws mus)))
        mean (e identity)
        c (Math/exp (- (dist/lgamma 1.0) (dist/lgamma 1.5)))]
    {:mu mean
     :mu-sd (Math/sqrt (- (e #(* % %)) (* mean mean)))
     :log-evidence (+ z (Math/log h))
     :sigmas (mapv (fn [y] (e (fn [mu] (* c (Math/sqrt (+ 1.0 (/ (* (- y mu) (- y mu)) 2.0)))))))
                   measurements)}))

;; SMC from the prior draws μ from N(0, 50²) — almost never near 9.5 — and
;; resampling at each observation copies whatever explains the data so far.
;; The program can do better by naming **guides**: μ from a Student-t around
;; the median measurement, and each precision from its exact conditional
;; given μ and the measurement. Every site is still weighted by p/q, so the
;; target is unchanged; the guided program is run by importance sampling.

(defn median [v] (nth (sort v) (quot (count v) 2)))

(defn guided-scientists []
  (spin
   (let [mu (sample (dist/normal 0.0 50.0) :id :mu
                    :proposal (dist/student-t 3.0 (median measurements) 1.0))]
     (loop [i 0 sigmas []]
       (if (< i 7)
         (let [y (nth measurements i)
               rate (+ 1.0 (/ (* (- y mu) (- y mu)) 2.0))
               precision (sample (dist/gamma 1.0 1.0) :id [:precision i]
                                 :proposal (dist/gamma 1.5 (/ 1.0 rate)))
               sigma (/ 1.0 (Math/sqrt precision))]
           (observe (dist/normal mu sigma) y :id [:y i])
           (recur (inc i) (conj sigmas sigma)))
         {:mu mu :sigmas sigmas})))))

(def scientist-runs
  (array-map
   "SMC, prior" (run 4 #(infer/infer (scientists) {:method :smc :particles 2000}))
   "importance, guided" (run 4 #(infer/infer (guided-scientists) {:method :importance :particles 2000}))))

(kind/table
 {:column-names (into ["" "E[μ]" "sd μ"] (concat (map #(str "E[σ" (inc %) "]") (range 7))
                                                 ["log evidence" "distinct μ" "ms"]))
  :row-vectors (into [(into ["exact" (r3 (:mu scientists-exact)) (r3 (:mu-sd scientists-exact))]
                            (concat (map r3 (:sigmas scientists-exact))
                                    [(r3 (:log-evidence scientists-exact)) "–" "–"]))]
                     (for [[label {:keys [measure ms]}] scientist-runs]
                       (let [mu (diagnostics/summary measure :mu)]
                         (into [label (r3 (:mean mu)) (r3 (:sd mu))]
                               (concat (for [i (range 7)]
                                         (r3 (:mean (diagnostics/summary measure #(nth (:sigmas %) i)))))
                                       [(r3 (m/log-marginal measure))
                                        (diagnostics/distinct-count measure :mu) ms])))))})

;; The diagnostic is the number of distinct values of μ among the final
;; particles (`diagnostics/distinct-count`): the histories that survived
;; resampling. SMC from the prior ends with its particles descended from a
;; handful of draws of μ, or one, and its evidence is far off; the guided
;; importance sampler keeps every draw, and its weight-based ESS says how
;; much they are worth:

(m/effective-sample-size (:measure (get scientist-runs "importance, guided")))

;; ## 4. A Gaussian mixture and label switching
;;
;; Six points from two clusters, each point assigned to cluster 1 or 2 with
;; probability ½, cluster means μ₁, μ₂ ~ N(0, 3²), unit noise. The labels
;; are arbitrary: swapping μ₁ and μ₂ (and every assignment) gives the same
;; density, so the posterior has two mirror-image modes.

(def points [-2.2 -1.7 -2.5 1.8 2.4 1.6])

(defn mixture []
  (spin
   (let [mu1 (sample (dist/normal 0.0 3.0) :id :mu1)
         mu2 (sample (dist/normal 0.0 3.0) :id :mu2)]
     (loop [i 0]
       (when (< i (count points))
         (let [z (sample (dist/bernoulli 0.5) :id [:z i])]
           (observe (dist/normal (if (= 1 z) mu2 mu1) 1.0) (nth points i) :id [:y i]))
         (recur (inc i))))
     {:mu1 mu1 :mu2 mu2 :low (min mu1 mu2) :high (max mu1 mu2)})))

;; Exact, by enumerating the 2⁶ assignments: given an assignment the two
;; means are independent normals (conjugate), with a closed-form marginal
;; likelihood, and the expected maximum of two independent normals is
;; closed-form too. By symmetry P(μ₁ < μ₂) = ½ and E[μ₁] = E[μ₂].

(def mixture-exact
  (let [group (fn [v]
                (let [k (count v) s (reduce + v) precision (+ k (/ 1.0 9))]
                  {:mean (/ s precision) :var (/ 1.0 precision)
                   :log-ml (- (* -0.5 k (Math/log (* 2 Math/PI)))
                              (* 0.5 (Math/log (+ 1 (* 9 k))))
                              (* 0.5 (- (reduce + (map * v v)) (/ (* 9 s s) (+ 1 (* 9 k))))))}))
        std (dist/normal 0.0 1.0)
        terms (for [a (range (bit-shift-left 1 (count points)))]
                (let [g1 (group (keep-indexed (fn [i y] (when-not (bit-test a i) y)) points))
                      g2 (group (keep-indexed (fn [i y] (when (bit-test a i) y)) points))
                      s (Math/sqrt (+ (:var g1) (:var g2)))
                      d (/ (- (:mean g2) (:mean g1)) s)
                      high (+ (* (:mean g2) (dist/cdf std d)) (* (:mean g1) (- 1 (dist/cdf std d)))
                              (* s (Math/exp (dist/logpdf std d))))]
                  {:log-w (+ (:log-ml g1) (:log-ml g2))
                   :mu1-lower (dist/cdf std d)
                   :mu1 (:mean g1)
                   :high high
                   :low (- (+ (:mean g1) (:mean g2)) high)}))
        z (m/log-sum-exp (map :log-w terms))
        e (fn [k] (reduce + (map #(* (Math/exp (- (:log-w %) z)) (k %)) terms)))]
    {:mu1-lower (e :mu1-lower) :mu1 (e :mu1) :low (e :low) :high (e :high)}))

(def mixture-runs
  (array-map
   "random-walk MH, 4 chains" (run 5 #(infer/infer (mixture) {:method :rmh :iterations 3000 :burn 500 :step-size 0.5}))
   "SMC" (run 5 #(infer/infer (mixture) {:method :smc :particles 2000}))))

(defn mu1-lower [v] (if (< (:mu1 v) (:mu2 v)) 1.0 0.0))

(def mixture-summaries
  (update-vals mixture-runs (fn [{:keys [measure]}]
                              (into {} (for [k [:mu1 :low :high]] [k (diagnostics/summary measure k)])))))

(kind/table
 {:column-names ["" "P(μ₁ < μ₂)" "E[μ₁]" "E[lower mean]" "E[upper mean]" "ms"]
  :row-vectors (into [["exact" (r3 (:mu1-lower mixture-exact)) (r3 (:mu1 mixture-exact))
                       (r3 (:low mixture-exact)) (r3 (:high mixture-exact)) "–"]]
                     (for [[label {:keys [measure ms]}] mixture-runs]
                       [label (r3 (get (distribution measure mu1-lower) 1.0 0.0))
                        (r3 (get-in mixture-summaries [label :mu1 :mean]))
                        (r3 (get-in mixture-summaries [label :low :mean]))
                        (r3 (get-in mixture-summaries [label :high :mean])) ms]))})

;; Pooled, the chains may look right. Chain by chain they are not: each
;; finds one labelling and keeps it, since swapping the labels needs every
;; assignment and both means to change at once. P(μ₁ < μ₂) within each
;; chain:

(let [measure (:measure (get mixture-runs "random-walk MH, 4 chains"))]
  (mapv (fn [c] (r3 (/ (reduce + c) (count c)))) (diagnostics/chains measure mu1-lower)))

;; R-hat sees it: far above 1 on μ₁, whose chains disagree, and much closer
;; to 1 on the label-invariant lower and upper means, which do not care
;; which labelling a chain found:

(kind/table
 {:column-names ["quantity" "R-hat" "bulk ESS"]
  :row-vectors (for [k [:mu1 :low :high]]
                 (let [s (get-in mixture-summaries ["random-walk MH, 4 chains" k])]
                   [(name k) (r3 (:rhat s)) (r3 (:ess-bulk s))]))})

;; SMC draws μ₁ and μ₂ from the prior before any data, so both labellings
;; are in the population from the start, and resampling keeps both. Its
;; draws of μ₁ show the two modes:

(-> (tc/dataset {:mu1 (infer/predict (:measure (get mixture-runs "SMC")) (comp :mu1 m/get-value) 2000)})
    (plotly/layer-histogram {:=x :mu1 :=histogram-nbins 60}))

;; A label-switching posterior is not a failure of the model; ask
;; label-invariant questions (the lower mean, whether two points share a
;; cluster), or break the symmetry with an ordering constraint.

;; ## 5. Branching
;;
;; Anglican's AISTATS 2014 benchmark: r ~ Poisson(4); if r > 4 the rate is
;; 6, otherwise it is 1 + fib(3r) plus a second Poisson(4) draw; one count,
;; 6, is observed. The program's structure depends on r: the second draw
;; exists only on one branch. Its sites have no `:id`.

(defn- fib [n] (loop [a 0 b 1 k 0] (if (= k n) a (recur b (+ a b) (inc k)))))

(defn branching []
  (spin
   (let [count-prior (dist/poisson 4)
         r (sample count-prior)
         l (if (< 4 r)
             6
             (+ 1 (fib (* 3 r)) (sample count-prior)))]
     (observe (dist/poisson l) 6)
     r)))

;; Exact by enumeration over r and the second draw (both truncated far into
;; the tail):

(def branching-exact
  (let [cp (dist/poisson 4)
        joint (for [r (range 40)]
                [r (if (< 4 r)
                     (+ (dist/logpdf cp r) (dist/logpdf (dist/poisson 6) 6))
                     (m/log-sum-exp
                      (for [s (range 60)]
                        (+ (dist/logpdf cp r) (dist/logpdf cp s)
                           (dist/logpdf (dist/poisson (+ 1 (fib (* 3 r)) s)) 6)))))])
        z (m/log-sum-exp (map second joint))]
    {:posterior (into {} (map (fn [[r lp]] [r (Math/exp (- lp z))]) joint))
     :log-evidence z}))

(def branching-runs
  (array-map
   "SMC" (run 6 #(infer/infer (branching) {:method :smc :particles 4000}))
   "single-site MH, 4 chains" (run 6 #(infer/infer (branching) {:method :mh :iterations 2000 :burn 200}))))

(kind/table
 {:column-names (into ["" "total variation"] (concat (map #(str "P(r = " % ")") (range 8)) ["ms"]))
  :row-vectors (into [(into ["exact" 0.0] (concat (map #(r3 (get-in branching-exact [:posterior %])) (range 8)) ["–"]))]
                     (for [[label {:keys [measure ms]}] branching-runs]
                       (let [d (distribution measure identity)]
                         (into [label (r3 (total-variation d (:posterior branching-exact)))]
                               (concat (map #(r3 (get d % 0.0)) (range 8)) [ms])))))})

;; Diagnostics: SMC's evidence against the exact one, and the chains' R-hat
;; and ESS for r.

{:smc-log-evidence (r3 (m/log-marginal (:measure (get branching-runs "SMC"))))
 :exact-log-evidence (r3 (:log-evidence branching-exact))
 :mh-r (let [cs (diagnostics/chains (:measure (get branching-runs "single-site MH, 4 chains")) identity)]
         {:rhat (r3 (diagnostics/rhat cs)) :ess-bulk (r3 (diagnostics/ess-bulk cs))})}

;; **Structural addresses.** Without `:id`, a site's address is its place
;; in the program, not its position in the run, so r keeps its address
;; whether or not the second draw happens — which is what lets single-site
;; MH change r and jump between branches. A run on each branch:

(let [traces (map (comp m/get-trace first) (m/get-particles (:measure (get branching-runs "SMC"))))
      two-sites (first (filter #(= 2 (count %)) traces))
      three-sites (first (filter #(= 3 (count %)) traces))]
  (kind/table
   {:column-names ["run" "address" "value" "observed?"]
    :row-vectors (for [[label t] [["r > 4" two-sites] ["r ≤ 4" three-sites]]
                       [address {:keys [value observed?]}] (sort-by (comp str key) t)]
                   [label (str address) value observed?])}))

;; ## 6. Coal-mining disasters
;;
;; British coal-mining disasters per year, 1851–1962 (Jarrett 1979; the 191
;; dates of R's `boot::coal`, counted by year). The rate fell at some point
;; — the classic change-point model puts one switch year s, uniform, between
;; an early and a late Poisson rate, each with an Exponential(1) prior.

(def disasters
  [4 5 4 1 0 4 3 4 0 6 3 3 4 0 2 6 3 3 5 4 5 3 1 4 4 1 5 5 3 4 2 5 2 2 3 4 2 1
   3 2 2 1 1 1 1 3 0 0 1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1 0 1 0 1 0 0 0 2 1 0 0 0
   1 1 0 2 3 3 1 1 2 1 1 1 1 2 3 3 0 0 0 1 4 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1])

(def n-years (count disasters))

;; With 112 observations of two static rates, SMC with an observe site per
;; year resamples the rates' early draws away; the model instead scores the
;; data in one `factor`, its log-likelihood, computed in plain Clojure (a
;; `reduce` is fine outside the sites).

(defn coal-log-likelihood [s early late]
  (reduce + (map-indexed (fn [t c] (dist/logpdf (dist/poisson (if (< t s) early late)) c)) disasters)))

(defn coal []
  (spin
   (let [s (sample (dist/uniform-discrete 1 n-years) :id :switch)
         early (sample (dist/exponential 1.0) :id :early)
         late (sample (dist/exponential 1.0) :id :late)]
     (factor (coal-log-likelihood s early late))
     {:switch s :early early :late late})))

;; Exact: an Exponential(1) rate is Gamma(1, 1), conjugate to the Poisson,
;; so for each switch year both rates integrate out in closed form —
;; a segment of m years with C disasters contributes
;; Γ(1 + C) (m + 1)^−(1 + C) / Πcᵢ! — and the posterior over the 111 switch
;; years is a sum over them. Given s, E[early] = (1 + C_early)/(s + 1).

(def coal-exact
  (let [segment (fn [cs] (let [m (count cs) c (reduce + cs)]
                           (- (dist/lgamma (+ 1.0 c)) (* (+ 1.0 c) (Math/log (+ m 1.0)))
                              (reduce + (map #(dist/lgamma (+ 1.0 %)) cs)))))
        lps (into {} (for [s (range 1 n-years)]
                       [s (+ (segment (subvec disasters 0 s)) (segment (subvec disasters s)))]))
        z (m/log-sum-exp (vals lps))
        posterior (into {} (map (fn [[s lp]] [s (Math/exp (- lp z))]) lps))]
    {:posterior posterior
     :early (reduce + (for [[s p] posterior] (* p (/ (+ 1.0 (reduce + (subvec disasters 0 s))) (+ s 1.0)))))
     :late (reduce + (for [[s p] posterior]
                       (* p (/ (+ 1.0 (reduce + (subvec disasters s))) (+ (- n-years s) 1.0)))))
     :log-evidence (- z (Math/log (dec n-years)))}))

;; Tempered SMC moves the population from the prior to the posterior through
;; tempered targets, with Metropolis–Hastings moves at each temperature; the
;; Markov chains move the switch year by prior proposals and the rates by a
;; random walk.

(def coal-runs
  (array-map
   "tempered SMC" (run 7 #(infer/infer (coal) {:method :tempered :particles 1000 :moves 3}))
   "random-walk MH, 4 chains" (run 7 #(infer/infer (coal) {:method :rmh :iterations 3000 :burn 1000 :step-size 0.3}))))

(defn year [s] (+ 1851 s))

(def coal-summaries
  (update-vals coal-runs (fn [{:keys [measure]}]
                           (into {} (for [k [:early :late]] [k (diagnostics/summary measure k)])))))

(kind/table
 {:column-names ["" "E[switch year]" "P(most likely year)" "total variation" "E[early]" "E[late]" "ms"]
  :row-vectors (let [mode (key (apply max-key val (:posterior coal-exact)))]
                 (into [["exact" (r3 (year (reduce + (map (fn [[s p]] (* s p)) (:posterior coal-exact)))))
                         (str (year mode) ": " (r3 (get-in coal-exact [:posterior mode]))) 0.0
                         (r3 (:early coal-exact)) (r3 (:late coal-exact)) "–"]]
                       (for [[label {:keys [measure ms]}] coal-runs]
                         (let [d (distribution measure :switch)]
                           [label (r3 (year (reduce + (map (fn [[s p]] (* s p)) d))))
                            (r3 (get d mode 0.0))
                            (r3 (total-variation d (:posterior coal-exact)))
                            (r3 (get-in coal-summaries [label :early :mean]))
                            (r3 (get-in coal-summaries [label :late :mean])) ms]))))})

;; The switch year is the first year of the late regime. The posterior over
;; it, exact and estimated:

(-> (tc/dataset
     (concat (for [[s p] (:posterior coal-exact) :when (> p 1e-3)]
               {:year (year s) :probability p :source "exact"})
             (for [[label {:keys [measure]}] coal-runs
                   [s p] (distribution measure :switch) :when (> p 1e-3)]
               {:year (year s) :probability p :source label})))
    (plotly/layer-bar {:=x :year :=y :probability :=color :source}))

;; Diagnostics: the tempered sampler's evidence against the exact one and
;; how many distinct switch years its particles hold, and the chains' R-hat
;; and ESS. For the discrete switch year, R-hat and the
;; bulk ESS only (`diagnostics/rhat`, `diagnostics/ess-bulk`): the tail ESS
;; counts the draws beyond the 5% and 95% quantiles, which a quantity on a
;; few values may not have.
;;
;; The switch year is the hard part for tempered SMC. Its particles cover
;; the plausible years, but their proportions are noisy: a move proposes a
;; discrete site from the prior — one of 111 years, of which only a few
;; are plausible — so few such moves are accepted, and resampling decides
;; the rest. The rates, moved by a random walk, come out well. More
;; particles, or more moves per temperature (`:moves`), buy a better
;; switch-year posterior.

{:tempered-log-evidence (r3 (m/log-marginal (:measure (get coal-runs "tempered SMC"))))
 :tempered-distinct-switch-years (diagnostics/distinct-count (:measure (get coal-runs "tempered SMC")) :switch)
 :exact-log-evidence (r3 (:log-evidence coal-exact))
 :chains (into {:switch (let [cs (diagnostics/chains (:measure (get coal-runs "random-walk MH, 4 chains")) :switch)]
                          {:rhat (r3 (diagnostics/rhat cs)) :ess-bulk (r3 (diagnostics/ess-bulk cs))})}
               (for [k [:early :late]]
                 [k (update-vals (select-keys (get-in coal-summaries ["random-walk MH, 4 chains" k])
                                              [:rhat :ess-bulk :ess-tail])
                                 r3)]))}

;; ## 7. Eight schools
;;
;; Coaching effects on a test in eight schools, each estimated with a known
;; standard error (Rubin 1981): y = 28, 8, −3, 7, −1, 1, 18, 12 and
;; σ = 15, 10, 16, 11, 9, 11, 10, 18. A hierarchical model pools them: each
;; school's effect θⱼ ~ N(μ, τ), with μ ~ N(0, 5²) and τ ~ half-Cauchy(5).

(def school-y [28.0 8.0 -3.0 7.0 -1.0 1.0 18.0 12.0])
(def school-sd [15.0 10.0 16.0 11.0 9.0 11.0 10.0 18.0])

;; The **centred** parameterisation samples θⱼ directly:

(defn schools-centred []
  (spin
   (let [mu (sample (dist/normal 0.0 5.0) :id :mu)
         tau (sample (dist/half-cauchy 5.0) :id :tau)]
     (loop [j 0]
       (when (< j 8)
         (let [theta (sample (dist/normal mu tau) :id [:theta j])]
           (observe (dist/normal theta (nth school-sd j)) (nth school-y j) :id [:y j]))
         (recur (inc j))))
     {:mu mu :tau tau :log-tau (Math/log tau)})))

;; The **non-centred** one samples standardized effects ηⱼ ~ N(0, 1) and
;; sets θⱼ = μ + τηⱼ — the same model:

(defn schools-noncentred []
  (spin
   (let [mu (sample (dist/normal 0.0 5.0) :id :mu)
         tau (sample (dist/half-cauchy 5.0) :id :tau)]
     (loop [j 0]
       (when (< j 8)
         (let [eta (sample (dist/normal 0.0 1.0) :id [:eta j])]
           (observe (dist/normal (+ mu (* tau eta)) (nth school-sd j)) (nth school-y j) :id [:y j]))
         (recur (inc j))))
     {:mu mu :tau tau :log-tau (Math/log tau)})))

;; Reference: integrating out the θⱼ gives yⱼ ~ N(μ, √(σⱼ² + τ²)), so the
;; posterior of (μ, τ) is two-dimensional, and a grid computes its means
;; and τ's median (τ up to 100; the likelihood decays like τ⁻⁸ beyond).

(def schools-exact
  (let [mus (mapv #(+ -25.0 (* 0.1 %)) (range 601))
        taus (mapv #(* 0.05 (+ 0.5 %)) (range 2000))
        pm (dist/normal 0.0 5.0) pt (dist/half-cauchy 5.0)
        log-normal-density (fn [y mu v] (* -0.5 (+ (Math/log (* 2 Math/PI v)) (/ (* (- y mu) (- y mu)) v))))
        lp (fn [mu tau] (+ (dist/logpdf pm mu) (dist/logpdf pt tau)
                           (reduce + (map (fn [y s] (log-normal-density y mu (+ (* s s) (* tau tau))))
                                          school-y school-sd))))
        cells (vec (for [mu mus tau taus] [mu tau (lp mu tau)]))
        z (m/log-sum-exp (map #(nth % 2) cells))
        ws (mapv #(Math/exp (- (nth % 2) z)) cells)
        tau-marginal (reduce (fn [acc [w [_ tau]]] (update acc tau (fnil + 0.0) w))
                             (sorted-map) (map vector ws cells))]
    {:mu (reduce + (map (fn [w [mu]] (* w mu)) ws cells))
     :tau (reduce + (map (fn [w [_ tau]] (* w tau)) ws cells))
     :tau-median (loop [[[tau w] & more] (seq tau-marginal) acc 0.0]
                   (if (>= (+ acc w) 0.5) tau (recur more (+ acc w))))}))

;; Four random-walk chains on each, the same budget:

(def school-runs
  (array-map
   "centred" (run 8 #(infer/infer (schools-centred) {:method :rmh :iterations 4000 :burn 1000 :step-size 2.0}))
   "non-centred" (run 8 #(infer/infer (schools-noncentred) {:method :rmh :iterations 4000 :burn 1000 :step-size 2.0}))))

(def school-summaries
  (update-vals school-runs (fn [{:keys [measure]}]
                             (into {} (for [k [:mu :tau :log-tau]] [k (diagnostics/summary measure k)])))))

(kind/table
 {:column-names ["" "E[μ]" "E[τ]" "median τ" "R-hat μ" "R-hat log τ" "bulk ESS log τ" "tail ESS log τ" "ms"]
  :row-vectors (into [["grid" (r3 (:mu schools-exact)) (r3 (:tau schools-exact)) (r3 (:tau-median schools-exact))
                       "–" "–" "–" "–" "–"]]
                     (for [[label {:keys [ms]}] school-runs]
                       (let [{:keys [mu tau log-tau]} (get school-summaries label)]
                         [label (r3 (:mean mu)) (r3 (:mean tau)) (r3 (get-in tau [:quantiles :p50]))
                          (r3 (:rhat mu)) (r3 (:rhat log-tau)) (r3 (:ess-bulk log-tau)) (r3 (:ess-tail log-tau)) ms])))})

;; In the centred form, θⱼ and τ are tied: when τ is small every θⱼ must sit
;; close to μ, so a step in τ is rejected unless the θⱼ happen to fit — the
;; posterior is a funnel, and the chains get stuck in its neck or its mouth.
;; R-hat far above 1.01 and a handful of effective draws say so. The
;; non-centred form makes the ηⱼ independent of τ a priori and mixes far
;; better on the same budget, and its estimates approach the grid's — though
;; for real use it would run longer, to a bulk ESS of a few hundred. Gradient
;; methods suffer from the funnel too (Stan reports divergences); for HMC on
;; numerical blocks see [Blocks and HMC](foerster.blocks.html).
