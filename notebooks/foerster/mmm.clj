;; # Marketing Mix Modeling

;; How much did each advertising channel add to sales, and where should the
;; next euro go? A marketing mix model (MMM) answers both from weekly sales
;; and spend. It is the bread and butter of Bayesian modeling in marketing —
;; [PyMC-Marketing](https://www.pymc-marketing.io/), Google's
;; [Meridian](https://developers.google.com/meridian) and Meta's
;; [Robyn](https://facebookexperimental.github.io/Robyn/) are built around
;; it — and a good test of a probabilistic programming system: carryover and
;; saturation make it nonlinear, channels move together, and the answers
;; people act on are counterfactuals.
;;
;; This notebook fits the standard model to data simulated from known
;; parameters, so every estimate can be checked against the truth, and then
;; asks it what marketers ask: return on spend per channel, what a lift test
;; changes, and how to split a budget when the downside matters. Every
;; number is computed as the notebook renders, with fixed seeds.

(ns foerster.mmm
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as diagnostics]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

(def world (sp/create-execution-context))

(defn r3 [x] (/ (Math/round (* 1000.0 (double x))) 1000.0))

;; ## The model
;;
;; Weekly sales are a baseline plus what each channel's spend adds:
;;
;; - **carryover** (adstock): spend keeps working for a few weeks. The
;;   geometric adstock of channel c with retention α_c is a weighted average
;;   of this and the last seven weeks' spend, with weights α_c^lag;
;; - **saturation**: the response to adstocked spend a flattens,
;;   s(a) = (1 − e^(−λa)) / (1 + e^(−λa)), steeper for a larger λ_c;
;; - each channel's effect is β_c · s(adstock), with β_c > 0;
;; - the baseline is an intercept, a linear trend and yearly seasonality (two
;;   Fourier harmonics); sales scatter around the sum with standard deviation
;;   σ.
;;
;; That is PyMC-Marketing's default model (`GeometricAdstock(l_max=8)`,
;; `LogisticSaturation`), with its default priors: β ~ HalfNormal(2),
;; α ~ Beta(1, 3), λ ~ Gamma(3, 1), σ ~ HalfNormal(2), Normal(0, 2) on the
;; intercept and the seasonal coefficients, Normal(0, 1) on the trend. Spend
;; is scaled to [0, 1] per channel and sales by their maximum, as there.

(def weeks 104)
(def channels 3)
(def l-max 8)

(defn adstock
  "Geometric adstock of `x` (a double array) with retention `alpha`,
  normalized: Σ_l α^l x[t−l] / Σ_l α^l over l < l-max."
  ^doubles [^doubles x alpha]
  (let [n (alength x)
        alpha (double alpha)
        out (double-array n)
        norm (loop [l 0 w 1.0 acc 0.0] (if (= l l-max) acc (recur (inc l) (* w alpha) (+ acc w))))]
    (dotimes [t n]
      (loop [l 0 w 1.0 acc 0.0]
        (if (or (= l l-max) (> l t))
          (aset out t (/ acc norm))
          (recur (inc l) (* w alpha) (+ acc (* w (aget x (- t l))))))))
    out))

(defn saturate ^double [lam ^double a]
  (let [e (Math/exp (- (* lam a)))] (/ (- 1.0 e) (+ 1.0 e))))

(defn fourier
  "[sin 2πt/52, cos 2πt/52, sin 4πt/52, cos 4πt/52] of week t."
  [t]
  (let [w (/ (* 2.0 Math/PI t) 52.0)]
    [(Math/sin w) (Math/cos w) (Math/sin (* 2 w)) (Math/cos (* 2 w))]))

(def ^:private fourier-table
  (let [n 1000]
    (vec (for [k (range 4)] (double-array (map #(nth (fourier %) k) (range n)))))))

(defn mean-sales
  "μ_t for the parameters `p` ({:intercept :trend :beta :alpha :lam :gamma})
  and spend `xs` (one double array per channel)."
  ^doubles [{:keys [intercept trend beta alpha lam gamma]} xs]
  (let [n (alength ^doubles (first xs))
        out (double-array n)
        intercept (double intercept) trend (double trend)]
    (dotimes [t n]
      (aset out t (+ intercept (* trend (/ t (double n))))))
    (dotimes [k 4]
      (let [g (double (nth gamma k)) ^doubles f (nth fourier-table k)]
        (dotimes [t n] (aset out t (+ (aget out t) (* g (aget f t)))))))
    (dotimes [c (count xs)]
      (let [^doubles a (adstock (nth xs c) (nth alpha c))
            b (double (nth beta c)) lam (double (nth lam c))]
        (dotimes [t n] (aset out t (+ (aget out t) (* b (saturate lam (aget a t))))))))
    out))

;; ## Data with a known answer
;;
;; Two years of weekly spend on three channels: the first in campaigns, the
;; second steady with a seasonal swing, and the third — the hard case —
;; largely following the first (correlation about 0.8), as when two
;; channels are booked together. Sales come from the model above with these
;; parameters:

(def truth
  {:intercept 0.35 :trend 0.1
   :beta [0.35 0.25 0.1] :alpha [0.5 0.2 0.3] :lam [4.0 3.0 3.5]
   :gamma [0.04 0.02 -0.01 0.015] :sigma 0.03})

(def spend
  (let [rng (java.util.Random. 7)
        u #(.nextDouble rng)
        x1 (vec (for [t (range weeks)] (if (< (mod t 13) 4) (+ 0.6 (* 0.4 (u))) (* 0.15 (u)))))
        x2 (vec (for [t (range weeks)] (+ 0.5 (* 0.3 (Math/sin (/ (* 2 Math/PI t) 52.0))) (* 0.2 (u)))))
        x3 (vec (map (fn [v] (max 0.0 (+ (* 0.8 v) (* 0.25 (u))))) x1))
        scale (fn [v] (let [mx (reduce max v)] (double-array (map #(/ % mx) v))))]
    [(scale x1) (scale x2) (scale x3)]))

(def sales
  (let [rng (java.util.Random. 8)
        mu (mean-sales truth spend)]
    (double-array (map #(+ % (* (:sigma truth) (.nextGaussian rng))) mu))))

(-> (tc/dataset {:week (range weeks)
                 :sales (seq sales)
                 :channel-1 (seq (first spend))
                 :channel-3 (seq (nth spend 2))})
    (tc/pivot->longer [:sales :channel-1 :channel-3] {:target-columns :series :value-column-name :value})
    (plotly/layer-line {:=x :week :=y :value :=color :series}))

;; ## The block
;;
;; The parameters form one block (see [blocks](foerster.blocks.html)): sixteen
;; numbers in their own coordinates — β, λ and σ positive, α in (0, 1) —
;; which foerster maps to unconstrained ones for NUTS. The density and its
;; gradient are written in plain Clojure: given the residuals, every
;; derivative is a sum over the weeks, except those through the adstock's
;; retention α, which are central differences of the channel's effect. (A
;; compiled gradient of any density is `foerster-raster`'s `defdensity`.)

(defn params
  "The parameter map of θ (natural coordinates, as the block sees them)."
  [^doubles th]
  {:intercept (aget th 0) :trend (aget th 1)
   :beta [(aget th 2) (aget th 3) (aget th 4)]
   :alpha [(aget th 5) (aget th 6) (aget th 7)]
   :lam [(aget th 8) (aget th 9) (aget th 10)]
   :gamma [(aget th 11) (aget th 12) (aget th 13) (aget th 14)]
   :sigma (aget th 15)})

(def half-normal-2 (dist/half-normal 2.0))
(def alpha-prior (dist/beta 1.0 3.0))
(def lam-prior (dist/gamma 3.0 1.0))
(def normal-2 (dist/normal 0.0 2.0))
(def normal-1 (dist/normal 0.0 1.0))

(defn log-prior [{:keys [intercept trend beta alpha lam gamma sigma]}]
  (+ (dist/logpdf normal-2 intercept) (dist/logpdf normal-1 trend)
     (reduce + (map #(dist/logpdf half-normal-2 %) beta))
     (reduce + (map #(dist/logpdf alpha-prior %) alpha))
     (reduce + (map #(dist/logpdf lam-prior %) lam))
     (reduce + (map #(dist/logpdf normal-2 %) gamma))
     (dist/logpdf half-normal-2 sigma)))

(defn log-prior-gradient
  "∂ log-prior / ∂θ, in θ's order."
  [{:keys [intercept trend beta alpha lam gamma sigma]}]
  (vec (concat [(/ (- intercept) 4.0) (- trend)]
               (map #(/ (- %) 4.0) beta)            ; HalfNormal(2)
               (map #(/ -2.0 (- 1.0 %)) alpha)       ; Beta(1, 3)
               (map #(- (/ 2.0 %) 1.0) lam)          ; Gamma(3, 1)
               (map #(/ (- %) 4.0) gamma)
               [(/ (- sigma) 4.0)])))

(defn lift-term
  "A lift test's log density and its gradient in (β_c, λ_c): an experiment
  that moved channel c's steady weekly spend from x to x + dx measured an
  increase `delta` in weekly sales, with standard error `sd`. At steady
  spend the adstock is the spend, so the model's increase is
  β_c (s(λ_c (x + dx)) − s(λ_c x))."
  [{:keys [beta lam]} {:keys [channel x dx delta sd]}]
  (let [b (nth beta channel) l (nth lam channel)
        ds (fn [a] (let [e (Math/exp (- (* l a)))] (/ (* 2.0 e) (* (+ 1.0 e) (+ 1.0 e)))))
        gain (- (saturate l (+ x dx)) (saturate l x))
        model-delta (* b gain)
        z (/ (- delta model-delta) sd)
        r (/ z sd)]
    {:log-lik (- (* -0.5 z z) (Math/log sd) 0.9189385332046727)
     :d-beta (* r gain)
     :d-lam (* r b (- (* (+ x dx) (ds (+ x dx))) (* x (ds x))))}))

(defn value+grad
  "[log density, gradient] at θ (natural coordinates)."
  [^doubles th {:keys [xs ^doubles ys lift]}]
  (let [{:keys [beta alpha lam sigma] :as p} (params th)
        n (alength ys)
        mu (mean-sales p xs)
        s2 (* sigma sigma)
        ;; r_t = ∂ log-lik / ∂μ_t
        r (double-array n)
        _ (dotimes [t n] (aset r t (/ (- (aget ys t) (aget mu t)) s2)))
        sum-r (fn [f] (loop [t 0 acc 0.0] (if (= t n) acc (recur (inc t) (+ acc (* (aget r t) (double (f t))))))))
        ssr (loop [t 0 acc 0.0] (if (= t n) acc (let [e (- (aget ys t) (aget mu t))] (recur (inc t) (+ acc (* e e))))))
        log-lik (- (/ ssr (* -2.0 s2)) (* n (+ (Math/log sigma) 0.9189385332046727)))
        per-channel
        (for [c (range channels)]
          (let [x (nth xs c) b (nth beta c) a (nth alpha c) l (nth lam c)
                ^doubles ad (adstock x a)
                effect (fn [a'] (let [^doubles ad' (adstock x a')]
                                  (fn [t] (* b (saturate l (aget ad' t))))))
                h 1e-6
                up (effect (+ a h)) down (effect (- a h))]
            {:beta (sum-r #(saturate l (aget ad %)))
             :alpha (sum-r #(/ (- (up %) (down %)) (* 2.0 h)))
             :lam (sum-r #(let [z (* l (aget ad %)) e (Math/exp (- z))]
                            (* b (aget ad %) (/ (* 2.0 e) (* (+ 1.0 e) (+ 1.0 e))))))}))
        grad-lik (vec (concat [(sum-r (constantly 1.0)) (sum-r #(/ % (double n)))]
                              (map :beta per-channel) (map :alpha per-channel) (map :lam per-channel)
                              (for [k (range 4)] (let [^doubles f (nth fourier-table k)] (sum-r #(aget f %))))
                              [(- (/ ssr (* s2 sigma)) (/ n sigma))]))
        ;; a lift test is one more factor of β_c and λ_c (θ indices 2 + c, 8 + c)
        lt (when lift (lift-term p lift))
        c (:channel lift)
        grad (cond-> (mapv + grad-lik (log-prior-gradient p))
               lt (-> (update (+ 2 c) + (:d-beta lt))
                      (update (+ 8 c) + (:d-lam lt))))]
    [(+ (log-prior p) log-lik (if lt (:log-lik lt) 0.0))
     (double-array grad)]))

(defn pointwise
  "Each week's log density: {week log-lik}."
  [th {:keys [xs ys]}]
  (let [p (params th) mu (mean-sales p xs) law (fn [t] (dist/normal (aget mu t) (:sigma p)))]
    (into {} (for [t (range (alength ^doubles ys))] [t (dist/logpdf (law t) (aget ^doubles ys t))]))))

(def mmm
  (block/block {:block/id :mmm
                :block/coordinates :constrained
                :block/latents [{:name :intercept :shape []}
                                {:name :trend :shape []}
                                {:name :beta :shape [3] :support :positive}
                                {:name :alpha :shape [3] :support [:interval 0.0 1.0]}
                                {:name :lam :shape [3] :support :positive}
                                {:name :gamma :shape [4]}
                                {:name :sigma :shape [] :support :positive}]
                :block/target :complete-conditional}
               {:log-density (fn [th inputs] (first (value+grad th inputs)))
                :value+grad value+grad
                :pointwise pointwise
                :simulate (fn [th {:keys [xs ys]}]
                            (let [p (params th) mu (mean-sales p xs)]
                              (into {} (for [t (range (alength ^doubles ys))]
                                         [t (dist/draw (dist/normal (aget mu t) (:sigma p)))]))))}))

(def inputs {:xs spend :ys sales})

;; NUTS starts from a neutral point, θ = 0 in unconstrained coordinates
;; (β = λ = σ = 1, α = ½):

(def init (vec (concat [0.0 0.0] (repeat 3 0.0) (repeat 3 0.0) (repeat 3 0.0) (repeat 4 0.0) [0.0])))

(defn model [inputs]
  (spin (sample (block/block-dist mmm inputs) :id :theta :init init)))

;; ## Fitting
;;
;; NUTS, four chains of 1000 draws after 500 warm-up iterations, which adapt
;; the step size and a diagonal metric:

(defn run [seed f]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        out (sp/with-context world @(f))]
    {:measure out :ms (long (/ (- (System/nanoTime) t0) 1e6))}))

(def fit
  (run 1 #(infer/infer (model inputs) {:method :nuts :iterations 1500 :burn 500 :chains 4})))

(defn natural
  "The parameter map of a draw (θ in unconstrained coordinates)."
  [theta]
  (params (double-array (block/constrain mmm theta))))

(def parameter-rows
  (concat [[:intercept identity] [:trend identity]]
          (for [k [:beta :alpha :lam] c (range channels)] [k #(nth % c) (inc c)])
          [[:sigma identity]]))

(defn recovery-table [measure]
  (kind/table
   {:column-names ["parameter" "truth" "mean" "sd" "R-hat" "bulk ESS"]
    :row-vectors (for [[k f c] parameter-rows]
                   (let [s (diagnostics/summary measure #(f (get (natural %) k)))]
                     [(str (name k) (when c (str " " c))) (f (get truth k))
                      (r3 (:mean s)) (r3 (:sd s)) (r3 (:rhat s)) (long (:ess-bulk s))]))}))

(recovery-table (:measure fit))

;; The chains agree (R-hat at 1.00) and every parameter is recovered within
;; its posterior spread — but look at the third channel: its β is uncertain
;; over most of its prior. It spent when the first did, so the data cannot
;; tell their effects apart; the model says so instead of guessing.

;; ## Return on spend, by counterfactual
;;
;; A channel's contribution is what sales would have been without it: the
;; same model with that channel's spend set to zero. Its return on spend
;; (ROAS) is that difference summed over the two years, per unit of spend.
;; Computed for every posterior draw, it is a distribution:

(def draws
  (let [ps (m/get-particles (:measure fit))]
    (mapv (comp natural m/get-value first) (take-nth 10 ps))))

(defn without [xs c] (assoc xs c (double-array (alength ^doubles (nth xs c)))))

(defn roas [p xs c]
  (/ (reduce + (map - (mean-sales p xs) (mean-sales p (without xs c))))
     (reduce + (nth xs c))))

(defn interval [xs]
  (let [s (vec (sort xs)) n (count s)]
    [(nth s (long (* 0.05 n))) (nth s (long (* 0.95 n)))]))

(defn roas-table [draws]
  (kind/table
   {:column-names ["channel" "true ROAS" "posterior mean" "90% interval"]
    :row-vectors (for [c (range channels)]
                   (let [rs (map #(roas % spend c) draws)]
                     [(inc c) (r3 (roas truth spend c)) (r3 (/ (reduce + rs) (count rs)))
                      (mapv r3 (interval rs))]))}))

(roas-table draws)

;; ## The twin world
;;
;; "What would sales have been last quarter without channel 1?" has two
;; readings. The *interventional* one asks about a fresh quarter run without
;; it: the model's mean without channel 1, plus new noise. The
;; *counterfactual* one asks about the quarter that happened: its weeks had
;; their own shocks — a competitor's promotion, the weather — which the
;; observed sales and the model reveal as residuals (abduction), and which
;; would have happened without the ads too. In the twin world the residuals
;; stay and only the channel's effect is removed. (foerster's
;; [counterfactuals](foerster.counterfactuals.html) do this for any program whose
;; sites are mechanisms; with the observations inside a block, it is a line
;; of arithmetic.)
;;
;; Over weeks 92–103, with channel 1 off:

(def quarter (range 92 104))

(defn quarter-sum [^doubles v] (reduce + (map #(aget v %) quarter)))

(def twin
  (let [rng (java.util.Random. 9)]
    (vec (for [p draws]
           (let [mu (mean-sales p spend)
                 mu' (mean-sales p (without spend 0))
                 residual (- (quarter-sum sales) (quarter-sum mu))]
             {:counterfactual (+ (quarter-sum mu') residual)
              :interventional (+ (quarter-sum mu')
                                 (* (:sigma p) (Math/sqrt (count quarter)) (.nextGaussian rng)))})))))

(def true-counterfactual
  ;; the data's own noise is known here: it is how the data were made
  (+ (quarter-sum (mean-sales truth (without spend 0)))
     (- (quarter-sum sales) (quarter-sum (mean-sales truth spend)))))

(kind/table
 {:column-names ["" "observed" "truth" "mean" "sd" "90% interval"]
  :row-vectors (for [k [:counterfactual :interventional]]
                 (let [v (map k twin) mu (/ (reduce + v) (count v))]
                   [(name k) (r3 (quarter-sum sales)) (r3 true-counterfactual) (r3 mu)
                    (r3 (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) v)) (count v))))
                    (mapv r3 (interval v))]))})

;; Both contain the truth and are centred alike, but they answer different
;; questions, and the counterfactual is the sharper one: the quarter's
;; noise is no longer a source of uncertainty, only the parameters are.
;; Here the parameters dominate, so the gain is modest; with a better
;; identified channel, or a noisier quarter, it is larger. It is the answer
;; to "what did channel 1 do for us last quarter".

;; ## A lift test
;;
;; The collinear channel can be untangled by an experiment: raise its spend
;; in some regions for a few weeks and measure the lift. Suppose one raised
;; channel 3's steady weekly spend from 0.5 to 0.8 and measured a weekly
;; increase of 0.019 with standard error 0.003 (the truth is 0.018). The
;; experiment enters as one more factor in the block — the likelihood of the
;; measured lift given β₃ and λ₃ — as PyMC-Marketing adds lift tests:

(def lift {:channel 2 :x 0.5 :dx 0.3 :delta 0.019 :sd 0.003})

(def calibrated
  (run 2 #(infer/infer (model (assoc inputs :lift lift)) {:method :nuts :iterations 1500 :burn 500 :chains 4})))

(recovery-table (:measure calibrated))

(def calibrated-draws
  (let [ps (m/get-particles (:measure calibrated))]
    (mapv (comp natural m/get-value first) (take-nth 10 ps))))

(roas-table calibrated-draws)

;; One measured number halves the uncertainty of β₃ and, through the sales
;; the two channels share, narrows β₁'s too. Channel 3's ROAS over the two
;; years stays wide: it also depends on how fast the channel saturates (λ₃),
;; which a single spend step cannot pin down — a second step, or a test at
;; another spend level, would. An honest model tells you which experiment
;; is missing.

;; ## Splitting a budget
;;
;; Next quarter's weekly budget is 1.5 (in the scaled units), to be split
;; across the three channels. At steady spend w_c the adstock is w_c, so
;; the weekly sales the channels add are Σ_c β_c s(λ_c w_c). Each posterior
;; draw gives a response for each split; the expected response is their
;; mean, and the risk-averse criterion CVaR₁₀ is the mean of the worst
;; tenth of them — what the split delivers if the parameters are on the
;; unlucky side.

(defn response [{:keys [beta lam]} w] (reduce + (map #(* %1 (saturate %2 %3)) beta lam w)))

(def splits
  (for [a (range 0 31) b (range 0 (- 31 a))]
    (let [w [(* 0.05 a) (* 0.05 b) (* 0.05 (- 30 a b))]] w)))

(defn cvar [xs q] (let [s (sort xs) k (max 1 (long (* q (count s))))] (/ (reduce + (take k s)) k)))

(defn best-splits [draws]
  (let [scored (for [w splits]
                 (let [rs (map #(response % w) draws)]
                   {:split w :mean (/ (reduce + rs) (count rs)) :cvar (cvar rs 0.1)
                    :truth (response truth w)}))
        best-truth (apply max (map :truth scored))]
    (kind/table
     {:column-names ["criterion" "split (ch 1, 2, 3)" "expected" "CVaR₁₀" "true response" "regret"]
      :row-vectors (for [[label k] [["expected response" :mean] ["CVaR₁₀" :cvar] ["the truth (unknowable)" :truth]]]
                     (let [s (apply max-key k scored)]
                       [label (mapv r3 (:split s)) (r3 (:mean s)) (r3 (:cvar s)) (r3 (:truth s))
                        (r3 (- best-truth (:truth s)))]))})))

(best-splits draws)

;; Maximizing the expected response finds the truth's best split up to the
;; grid; the risk-averse split gives up about 1.5% of the true response to
;; spread spend across the channels whose response is uncertain, which
;; raises the worst-case tenth. With the lift test:

(best-splits calibrated-draws)

;; ## What else foerster brings to this
;;
;; - **Weekly updates without refitting.** Written with its weekly
;;   observations as sites, the model runs under SMC², which adds a week by
;;   reweighting and moving particles instead of refitting from scratch (see
;;   [streaming](foerster.streaming.html)); an always-on MMM is a stream.
;; - **Scenario worlds.** A budget scenario is a fork of the world the
;;   posterior lives in: each runs against the same model, data and
;;   connected systems, in isolation, and is discarded or kept.
;; - **Which experiment next.** The lift test above was chosen by hand. The
;;   one that most reduces the uncertainty of a decision is an expected
;;   information gain, a nested inference (`infer/conditional`) over the
;;   experiments one could run.
