(ns org.replikativ.foerster.posteriordb
  "Accuracy and speed against posteriordb's reference posteriors (Magnusson,
  Torgander, Bürkner, Zhang, Carpenter & Vehtari 2025).

  Not part of the test suite: a benchmark with explicit thresholds, run with
  `clojure -M:test:posteriordb [posterior …]` (all of them without
  arguments). It fetches each posterior's data and reference draws from
  posteriordb at a pinned commit into ~/.cache/foerster/posteriordb.

  Each posterior is one block written here from its Stan program, in
  unconstrained coordinates with the transforms' log-Jacobians, and with a
  numeric gradient (`block/with-numeric-gradient`). The methods that need a
  reference to start from — tempered SMC and parallel tempering — get the
  block's Laplace approximation with its covariance inflated
  `reference-inflation` times, which also makes posteriordb's flat priors
  proper to start from.

  For every posterior and method: per parameter, the error of the posterior
  mean in reference standard deviations and in Monte Carlo standard errors,
  and the ratio of the standard deviations, against the reference draws (10
  chains × 1000); the wall time; the smallest ESS (bulk ESS and R-hat for
  Markov chains, for SMC, which has none, the ESS its spread over `replicates` independent
  runs implies).

  A result is confident when every parameter's ESS is at least `min-ess`
  (400: Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021). A method whose
  run falls short is run again, longer (`grow`), up to `max-runs` times; a
  result that stays short fails as LOW-ESS. A parameter passes when its mean
  is within `mean-tolerance` reference sds and `z-tolerance` Monte Carlo
  standard errors (the run's and the reference's), and its sd ratio within
  1 ± `sd-tolerance`. At ESS 400 the Monte Carlo error of a mean is at most
  0.05 sd, so noise alone does not break the 0.2-sd tolerance, and a mean off
  by 4 standard errors is a bias, not a draw. SMC's own weight-based ESS
  overstates what resampled particles hold, so its standard error comes from
  `replicates` independent runs instead. Results go to
  experiments/results/posteriordb/ as EDN, printed next to the previous
  run's. Exits 1 if anything fails.

  `clojure -M:test:posteriordb control` runs the negative control instead
  (`control!`): a posterior with a known bug, which every method must fail."
  (:require [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.pprint :as pp]
            [clojure.string :as str]
            [org.replikativ.foerster.benchmark-test :refer [run-infer]]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.inference.optimize :as optimize]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def commit "91027f9cdd13b7e76283fec3f2898f59d4b83ec4")
(def mean-tolerance 0.2)
(def sd-tolerance 0.2)
(def z-tolerance 4.0)
(def min-ess 400)
(def ^:private max-runs 6)
(def ^:private reference-inflation 4.0)

;; --- posteriordb files ------------------------------------------------------

(def ^:private cache-dir
  (io/file (System/getProperty "user.home") ".cache" "foerster" "posteriordb" commit))

(defn fetch-file
  "The posteriordb file at `path` (relative to posterior_database/), from the
  cache or downloaded into it; a .zip holds one file, which is kept."
  [path]
  (let [zipped? (str/ends-with? path ".zip")
        f (io/file cache-dir (str/replace path #"\.zip$" ""))]
    (when-not (.exists f)
      (io/make-parents f)
      (let [tmp (io/file (str f ".part"))]
        (with-open [in (.openStream (java.net.URL. (str "https://raw.githubusercontent.com/stan-dev/posteriordb/"
                                                        commit "/posterior_database/" path)))]
          (if zipped?
            (with-open [z (java.util.zip.ZipInputStream. in)]
              (.getNextEntry z)
              (io/copy z tmp))
            (io/copy in tmp)))
        (.renameTo tmp f)))
    f))

(defn- fetch [path] (json/parse-string (slurp (fetch-file path))))

(defn posterior-info
  "posteriordb's description of `posterior`: its model_name, data_name, …"
  [posterior]
  (fetch (str "posteriors/" posterior ".json")))

(defn posterior-data [posterior]
  (fetch (str "data/data/" (get (posterior-info posterior) "data_name") ".json.zip")))

(defn stan-program
  "The Stan program of `posterior`, as text."
  [posterior]
  (slurp (fetch-file (str "models/stan/" (get (posterior-info posterior) "model_name") ".stan"))))

(defn reference-chains
  "{param [[draw …] …]}: the reference chains of `posterior`, by chain."
  [posterior]
  (let [chains (fetch (str "reference_posteriors/draws/draws/" posterior ".json.zip"))]
    (into (sorted-map)
          (for [param (keys (first chains))]
            [param (mapv (fn [c] (mapv double (get c param))) chains)]))))

(defn reference-draws
  "{param [draw …]}: the reference chains of `posterior`, pooled in one order
  for every parameter (draws at one index are one joint draw)."
  [posterior]
  (update-vals (reference-chains posterior) #(vec (apply concat %))))

(defn reference-stats
  "{param {:mean :sd :mcse}} of the pooled reference draws; the Monte Carlo
  standard error of the mean from the reference chains."
  [posterior]
  (into (sorted-map)
        (for [[param cs] (reference-chains posterior)
              :let [xs (vec (apply concat cs))
                    n (count xs)
                    mu (/ (reduce + xs) n)]]
          [param {:mean mu
                  :sd (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) xs)) (dec n)))
                  :mcse (d/mcse cs)}])))

;; --- densities --------------------------------------------------------------

(def ^:private half-log-2pi 0.9189385332046727)

(defn- sq ^double [^double x] (* x x))

(defn- normal-lp ^double [^double x ^double mu ^double s]
  (- (* -0.5 (sq (/ (- x mu) s))) (Math/log s) half-log-2pi))

(defn- cauchy-lp ^double [^double x ^double loc ^double s]
  (- (- (Math/log (* Math/PI s))) (Math/log1p (sq (/ (- x loc) s)))))

(defn- softplus ^double [^double t]
  (if (> t 0.0) (+ t (Math/log1p (Math/exp (- t)))) (Math/log1p (Math/exp t))))

(defn- sigmoid ^double [^double t] (/ 1.0 (+ 1.0 (Math/exp (- t)))))

(defn- log-sigmoid-jacobian
  "log of dx/dt for x = σ(t): log σ(t) + log(1 − σ(t))."
  ^double [^double t]
  (- (- (softplus (- t))) (softplus t)))

(defn- bernoulli-logit-lp ^double [^double y ^double eta]
  (- (* y eta) (softplus eta)))

(defn- darr [xs] (double-array (map double xs)))

(defn- indexed [prefix xs] (into {} (map-indexed (fn [i x] [(str prefix "[" (inc i) "]") x]) xs)))

(defn- linear-regression
  "y ~ N(Xβ, σ) with the given priors: θ = [β…, log σ]."
  [{:keys [prepare beta-prior sigma-prior]}]
  {:prepare prepare
   :log-density (fn [^doubles th {:keys [xs ^doubles y]}]
                  (let [k (count xs)
                        sigma (Math/exp (aget th k))
                        n (alength y)]
                    (loop [i 0 acc (+ (aget th k)
                                      (if sigma-prior (sigma-prior sigma) 0.0)
                                      (if beta-prior (reduce + (map #(beta-prior (aget th %)) (range k))) 0.0))]
                      (if (< i n)
                        (let [mu (loop [j 0 mu 0.0]
                                   (if (< j k) (recur (inc j) (+ mu (* (aget th j) (aget ^doubles (nth xs j) i)))) mu))]
                          (recur (inc i) (+ acc (normal-lp (aget y i) mu sigma))))
                        acc))))
   :params (fn [th {:keys [xs]}]
             (let [k (count xs)]
               (assoc (indexed "beta" (take k th)) "sigma" (Math/exp (nth th k)))))})

(defn- column [d k] (darr (get d k)))

(def ^:private models
  {"eight_schools-eight_schools_noncentered"
   ;; θ = [θ̃₁…θ̃₈, μ, log τ]; θ_j = μ + τ·θ̃_j
   {:prepare (fn [d] {:y (column d "y") :sigma (column d "sigma")})
    :log-density (fn [^doubles th {:keys [^doubles y ^doubles sigma]}]
                   (let [mu (aget th 8) tau (Math/exp (aget th 9))]
                     (loop [j 0 acc (+ (normal-lp mu 0.0 5.0) (cauchy-lp tau 0.0 5.0) (aget th 9))]
                       (if (< j 8)
                         (recur (inc j) (+ acc (normal-lp (aget th j) 0.0 1.0)
                                           (normal-lp (aget y j) (+ mu (* tau (aget th j))) (aget sigma j))))
                         acc))))
    :params (fn [th _]
              (let [mu (nth th 8) tau (Math/exp (nth th 9))]
                (assoc (indexed "theta" (map #(+ mu (* tau %)) (take 8 th))) "mu" mu "tau" tau)))}

   "kidiq-kidscore_momiq"
   (linear-regression {:prepare (fn [d] {:xs [(darr (repeat (get d "N") 1.0)) (column d "mom_iq")]
                                         :y (column d "kid_score")})
                       :sigma-prior #(cauchy-lp % 0.0 2.5)})

   "earnings-logearn_height"
   (linear-regression {:prepare (fn [d] {:xs [(darr (repeat (get d "N") 1.0)) (column d "height")]
                                         :y (darr (map #(Math/log %) (get d "earn")))})})

   "mesquite-logmesquite"
   (linear-regression {:prepare (fn [d]
                                  (let [lg (fn [k] (darr (map #(Math/log %) (get d k))))]
                                    {:xs [(darr (repeat (get d "N") 1.0)) (lg "diam1") (lg "diam2")
                                          (lg "canopy_height") (lg "total_height") (lg "density") (column d "group")]
                                     :y (lg "weight")}))})

   "sblrc-blr"
   (linear-regression {:prepare (fn [d] {:xs (vec (apply map (fn [& col] (darr col)) (get d "X")))
                                         :y (column d "y")})
                       :beta-prior #(normal-lp % 0.0 10.0)
                       :sigma-prior #(normal-lp % 0.0 10.0)})

   "arK-arK"
   ;; θ = [α, β₁…β_K, log σ]
   {:prepare (fn [d] {:k (get d "K") :y (column d "y")})
    :log-density (fn [^doubles th {:keys [k ^doubles y]}]
                   (let [k (long k) sigma (Math/exp (aget th (inc k))) n (alength y)
                         prior (+ (normal-lp (aget th 0) 0.0 10.0)
                                  (reduce + (map #(normal-lp (aget th (inc %)) 0.0 10.0) (range k)))
                                  (cauchy-lp sigma 0.0 2.5) (aget th (inc k)))]
                     (loop [t k acc prior]
                       (if (< t n)
                         (let [mu (loop [j 0 mu (aget th 0)]
                                    (if (< j k) (recur (inc j) (+ mu (* (aget th (inc j)) (aget y (- t (inc j)))))) mu))]
                           (recur (inc t) (+ acc (normal-lp (aget y t) mu sigma))))
                         acc))))
    :params (fn [th {:keys [k]}]
              (assoc (indexed "beta" (subvec (vec th) 1 (inc k)))
                     "alpha" (nth th 0) "sigma" (Math/exp (nth th (inc k)))))}

   "garch-garch11"
   ;; θ = [μ, log α₀, logit α₁, logit u], β₁ = (1 − α₁)·u
   {:prepare (fn [d] {:y (column d "y") :sigma1 (double (get d "sigma1"))})
    :log-density (fn [^doubles th {:keys [^doubles y sigma1]}]
                   (let [mu (aget th 0) a0 (Math/exp (aget th 1))
                         a1 (sigmoid (aget th 2)) b1 (* (- 1.0 a1) (sigmoid (aget th 3)))
                         n (alength y)
                         jac (+ (aget th 1) (log-sigmoid-jacobian (aget th 2))
                                (Math/log (- 1.0 a1)) (log-sigmoid-jacobian (aget th 3)))]
                     (loop [t 1 s (double sigma1) acc (+ jac (normal-lp (aget y 0) mu sigma1))]
                       (if (< t n)
                         (let [s (Math/sqrt (+ a0 (* a1 (sq (- (aget y (dec t)) mu))) (* b1 (sq s))))]
                           (recur (inc t) s (+ acc (normal-lp (aget y t) mu s))))
                         acc))))
    :params (fn [th _]
              (let [a1 (sigmoid (nth th 2))]
                {"mu" (nth th 0) "alpha0" (Math/exp (nth th 1)) "alpha1" a1
                 "beta1" (* (- 1.0 a1) (sigmoid (nth th 3)))}))}

   "low_dim_gauss_mix-low_dim_gauss_mix"
   ;; θ = [μ₁, log(μ₂ − μ₁), log σ₁, log σ₂, logit θ]
   {:prepare (fn [d] {:y (column d "y")})
    :log-density (fn [^doubles th {:keys [^doubles y]}]
                   (let [m1 (aget th 0) m2 (+ m1 (Math/exp (aget th 1)))
                         s1 (Math/exp (aget th 2)) s2 (Math/exp (aget th 3))
                         w (sigmoid (aget th 4)) lw (Math/log w) lw' (Math/log (- 1.0 w))
                         n (alength y)
                         prior (+ (normal-lp s1 0.0 2.0) (normal-lp s2 0.0 2.0)
                                  (normal-lp m1 0.0 2.0) (normal-lp m2 0.0 2.0)
                                  (* 4.0 lw) (* 4.0 lw')
                                  (aget th 1) (aget th 2) (aget th 3) (log-sigmoid-jacobian (aget th 4)))]
                     (loop [i 0 acc prior]
                       (if (< i n)
                         (let [a (+ lw (normal-lp (aget y i) m1 s1)) b (+ lw' (normal-lp (aget y i) m2 s2))
                               top (max a b)]
                           (recur (inc i) (+ acc top (Math/log (+ (Math/exp (- a top)) (Math/exp (- b top)))))))
                         acc))))
    :params (fn [th _]
              (let [m1 (nth th 0)]
                {"mu[1]" m1 "mu[2]" (+ m1 (Math/exp (nth th 1)))
                 "sigma[1]" (Math/exp (nth th 2)) "sigma[2]" (Math/exp (nth th 3))
                 "theta" (sigmoid (nth th 4))}))}

   "dogs-dogs"
   ;; θ = β; p = β₁ + β₂·avoidances + β₃·shocks so far
   {:prepare (fn [d]
               (let [rows (for [ys (get d "y")
                                :let [avoid (reductions + 0 (map #(- 1 %) ys))
                                      shock (reductions + 0 ys)]
                                [t y] (map-indexed vector ys)]
                            [(double y) (double (nth avoid t)) (double (nth shock t))])]
                 {:y (darr (map first rows)) :avoid (darr (map second rows)) :shock (darr (map #(nth % 2) rows))}))
    :log-density (fn [^doubles th {:keys [^doubles y ^doubles avoid ^doubles shock]}]
                   (let [n (alength y)]
                     (loop [i 0 acc (reduce + (map #(normal-lp (aget th %) 0.0 100.0) (range 3)))]
                       (if (< i n)
                         (recur (inc i) (+ acc (bernoulli-logit-lp (aget y i)
                                                                   (+ (aget th 0) (* (aget th 1) (aget avoid i))
                                                                      (* (aget th 2) (aget shock i))))))
                         acc))))
    :params (fn [th _] (indexed "beta" th))}

   "wells_data-wells_dist"
   {:prepare (fn [d] {:y (column d "switched") :dist (column d "dist")})
    :log-density (fn [^doubles th {:keys [^doubles y ^doubles dist]}]
                   (let [n (alength y)]
                     (loop [i 0 acc 0.0]
                       (if (< i n)
                         (recur (inc i) (+ acc (bernoulli-logit-lp (aget y i) (+ (aget th 0) (* (aget th 1) (aget dist i))))))
                         acc))))
    :params (fn [th _] (indexed "beta" th))}})

(def ^:private dimensions
  {"eight_schools-eight_schools_noncentered" 10 "kidiq-kidscore_momiq" 3 "earnings-logearn_height" 3
   "mesquite-logmesquite" 8 "sblrc-blr" 6 "arK-arK" 7 "garch-garch11" 4
   "low_dim_gauss_mix-low_dim_gauss_mix" 5 "dogs-dogs" 3 "wells_data-wells_dist" 2})

;; --- blocks -----------------------------------------------------------------

(defn- cholesky
  "The lower Cholesky factor of the symmetric positive definite `a` (vectors)."
  [a]
  (let [n (count a)]
    (reduce (fn [l [i j]]
              (let [s (reduce + (map #(* (get-in l [i %]) (get-in l [j %])) (range j)))]
                (assoc-in l [i j] (if (= i j)
                                    (Math/sqrt (- (get-in a [i i]) s))
                                    (/ (- (get-in a [i j]) s) (get-in l [j j]))))))
            (vec (repeat n (vec (repeat n 0.0))))
            (for [i (range n) j (range (inc i))] [i j]))))

(defn- gaussian-reference
  "{:sample :sample-log-density} of N(`mean`, `cov`)."
  [mean cov]
  (let [l (cholesky cov)
        n (count mean)
        log-det (* 2.0 (reduce + (map #(Math/log (get-in l [% %])) (range n))))
        ;; solve L z = x − mean by forward substitution
        whiten (fn [x] (reduce (fn [z i]
                                 (conj z (/ (- (- (nth x i) (nth mean i))
                                               (reduce + (map #(* (get-in l [i %]) (nth z %)) (range i))))
                                            (get-in l [i i]))))
                               [] (range n)))]
    {:sample (fn [_] (let [z (vec (repeatedly n #(dist/draw (dist/normal 0.0 1.0))))]
                       (mapv (fn [i] (+ (nth mean i) (reduce + (map #(* (get-in l [i %]) (nth z %)) (range (inc i))))))
                             (range n))))
     :sample-log-density (fn [th _] (let [z (whiten (vec th))]
                                      (- (* -0.5 (reduce + (map sq z))) (* 0.5 log-det) (* n half-log-2pi))))}))

(defn- make-block [posterior log-density extra]
  (block/block {:block/id (keyword posterior)
                :block/latents [{:name :theta :shape [(dimensions posterior)]}]
                :block/target :complete-conditional}
               (merge (block/with-numeric-gradient {:log-density log-density}) extra)))

(def ^:dynamic *reference*
  "How tempered SMC and parallel tempering get their reference: the block's
  `:laplace` approximation or its `:pathfinder` Gaussian, either with its
  covariance inflated `reference-inflation` times."
  :laplace)

(defn- setup
  "The block of `posterior` at its data, with a Gaussian reference (see
  `*reference*`). The samplers target `log-density` (default the
  posterior's); the reference always comes from the posterior's own."
  [posterior log-density]
  (let [{:keys [prepare] own :log-density} (models posterior)
        inputs (prepare (posterior-data posterior))
        d (block/block-dist (make-block posterior own nil) inputs)
        init (vec (repeat (dimensions posterior) 0.0))
        [mean cov] (case *reference*
                     :laplace (let [a (optimize/laplace d {:init init})] [(vec (:theta a)) (:covariance a)])
                     :pathfinder (let [a (optimize/pathfinder d {:init init})] [(:mean a) (:covariance a)]))
        reference (gaussian-reference mean (mapv (fn [row] (mapv #(* reference-inflation %) row)) cov))]
    {:block (make-block posterior (or log-density own) reference) :inputs inputs :mode mean}))

;; --- methods ----------------------------------------------------------------

(def ^:private method-options
  {:nuts {:method :nuts :iterations 2000 :chains 4 :burn 1000}
   :tempered {:method :tempered :particles 2000}
   :parallel-tempering {:method :parallel-tempering :chains 10 :rounds 11}})

(def ^:private grow
  "How a method runs longer when its ESS falls short of `min-ess`: twice the
  draws."
  {:nuts #(update % :iterations * 2)
   :tempered #(update % :particles * 2)
   :parallel-tempering #(update % :rounds inc)})

(defn- run-method
  ([posterior setup method] (run-method posterior setup method (method-options method)))
  ([posterior {:keys [block inputs mode]} _method options]
   (let [{:keys [params]} (models posterior)
         model (fn [] (spin (params (sample (block/block-dist block inputs) :id :theta :init mode) inputs)))
         t0 (System/nanoTime)
         measure (run-infer 1 #(infer/infer (model) options))]
     {:measure measure :seconds (/ (- (System/nanoTime) t0) 1e9)})))

(defn- chain-estimates
  "{param {:mean :sd :ess :mcse :rhat}} of a Markov-chain measure."
  [reference measure]
  (into {} (for [param (keys reference)
                 :let [s (d/summary measure #(get % param))]]
             [param {:mean (:mean s) :sd (:sd s) :ess (:ess-bulk s) :mcse (:mcse s) :rhat (:rhat s)}])))

(def ^:private replicates
  "Independent runs of a method without a Markov ESS (SMC): its Monte Carlo
  error is the spread of their means."
  16)

(defn- replicate-summary
  "{param {:mean :sd}} of one replicate's measure, so the measure need not be
  kept."
  [reference measure]
  (into {} (for [param (keys reference)]
             [param (select-keys (d/summary measure #(get % param)) [:mean :sd])])))

(defn- replicate-estimates
  "{param {:mean :sd :ess :mcse}} of `summaries` of independent runs: the
  mean of their means, its standard error from their spread, and the ESS
  that standard error implies."
  [reference summaries]
  (let [r (count summaries)]
    (into {} (for [param (keys reference)
                   :let [ss (mapv #(get % param) summaries)
                         means (mapv :mean ss)
                         mean (/ (reduce + means) r)
                         var-means (/ (reduce + (map #(let [e (- % mean)] (* e e)) means)) (dec r))
                         sd (Math/sqrt (/ (reduce + (map #(* (:sd %) (:sd %)) ss)) r))
                         mcse (Math/sqrt (/ var-means r))]]
               [param {:mean mean :sd sd :mcse mcse
                       :ess (if (pos? mcse) (/ (* sd sd) (* mcse mcse)) ##Inf)}]))))

(defn- compare-to [reference estimates]
  (into (sorted-map)
        (for [[param {ref-mean :mean ref-sd :sd ref-mcse :mcse}] reference
              :let [{:keys [ess mcse] :as s} (get estimates param)
                    err (- (:mean s) ref-mean)
                    z (/ err ref-sd)
                    z-mc (/ err (Math/sqrt (+ (* mcse mcse) (* ref-mcse ref-mcse))))
                    ratio (/ (:sd s) ref-sd)]]
          [param (cond-> {:mean (:mean s) :sd (:sd s) :ref-mean ref-mean :ref-sd ref-sd
                          :z z :z-mcse z-mc :sd-ratio ratio :ess ess :mcse mcse
                          :confident? (>= ess min-ess)
                          :pass? (and (>= ess min-ess)
                                      (< (Math/abs z) mean-tolerance)
                                      (< (Math/abs z-mc) z-tolerance)
                                      (< (Math/abs (- ratio 1.0)) sd-tolerance))}
                   (:rhat s) (assoc :rhat (:rhat s)))])))

(defn- run-once
  "One run of `method` with `options`: its estimates and seconds. SMC runs
  `replicates` times, seeded apart."
  [posterior setup reference method options]
  (if (= :tempered method)
    (let [results (mapv (fn [i]
                          (let [{:keys [measure seconds]}
                                (run-method posterior setup method (assoc options :seed (inc i)))]
                            {:seconds seconds :summary (replicate-summary reference measure)}))
                        (range replicates))]
      {:estimates (replicate-estimates reference (mapv :summary results))
       :seconds (reduce + (map :seconds results))})
    (let [{:keys [measure seconds]} (run-method posterior setup method options)]
      {:estimates (chain-estimates reference measure) :seconds seconds})))

(defn- run-confidently
  "Run `method` until every parameter's ESS reaches `min-ess`, growing the
  run (`grow`) up to `max-runs` runs. It stops growing when an ESS is not a
  number (a stuck chain: more draws do not help), and keeps the last
  completed run when a longer one fails (it timed out). The last run's
  comparison, options, run count and total seconds."
  [posterior setup reference method]
  (loop [options (method-options method) runs 1 total 0.0 previous nil]
    (let [attempt (try (run-once posterior setup reference method options)
                       (catch Exception e (if previous {::failed (ex-message e)} (throw e))))]
      (if (::failed attempt)
        (assoc previous :stopped (::failed attempt))
        (let [ps (compare-to reference (:estimates attempt))
              total (+ total (:seconds attempt))
              result {:params ps :seconds (:seconds attempt) :total-seconds total :options options :runs runs}]
          (if (or (every? :confident? (vals ps))
                  (= runs max-runs)
                  (some #(Double/isNaN (double (:ess %))) (vals ps)))
            result
            (recur ((grow method) options) (inc runs) total result)))))))

(defn- run-posterior [posterior log-density]
  (let [reference (reference-stats posterior)
        s (setup posterior log-density)]
    (vec (for [method (keys method-options)
               :let [{ps :params :keys [seconds total-seconds options runs stopped error]}
                     (try (run-confidently posterior s reference method)
                          (catch Exception e {:error (ex-message e)}))]]
           (if error
             {:posterior posterior :method method :error error :pass? false}
             (let [ess (map :ess (vals ps))
                   rhats (keep :rhat (vals ps))]
               (cond-> {:posterior posterior :method method :seconds seconds
                        :total-seconds total-seconds :options options :runs runs :stopped stopped
                        :max-z (reduce max (map (comp #(Math/abs %) :z) (vals ps)))
                        :max-z-mcse (reduce max (map (comp #(Math/abs %) :z-mcse) (vals ps)))
                        :max-sd-error (reduce max (map (comp #(Math/abs (- % 1.0)) :sd-ratio) (vals ps)))
                        :min-ess (reduce min ess)
                        :ess-per-second (/ (reduce min ess) seconds)
                        :confident? (every? :confident? (vals ps))
                        :pass? (every? :pass? (vals ps))
                        :params ps}
                 (seq rhats) (assoc :max-rhat (reduce max rhats)))))))))

;; --- results ----------------------------------------------------------------

(def ^:private results-dir (io/file "experiments" "results" "posteriordb"))

(defn- previous-results []
  (when-let [f (last (sort (filter #(str/ends-with? (.getName %) ".edn") (file-seq results-dir))))]
    {:file (.getName f) :rows (:results (read-string (slurp f)))}))

(defn- report [rows previous]
  (let [before (into {} (map (juxt (juxt :posterior :method) identity)) (:rows previous))
        fmt (fn [x spec] (if (number? x) (format spec (double x)) "—"))]
    (when previous (println "previous run:" (:file previous)))
    (pp/print-table
     [:posterior :method :pass :runs :seconds :before-s :max-z :max-z-mcse :max-sd-err :min-ess :ess/s]
     (for [{:keys [posterior method pass? confident? runs seconds max-z max-z-mcse max-sd-error min-ess
                   ess-per-second error]} rows
           :let [b (before [posterior method])]]
       {:posterior posterior :method (name method)
        :pass (cond error (str "ERROR " error) pass? "ok" (not confident?) "LOW-ESS" :else "FAIL")
        :runs (or runs "—")
        :seconds (fmt seconds "%.1f") :before-s (fmt (:seconds b) "%.1f")
        :max-z (fmt max-z "%.3f") :max-z-mcse (fmt max-z-mcse "%.2f") :max-sd-err (fmt max-sd-error "%.3f")
        :min-ess (fmt min-ess "%.0f") :ess/s (fmt ess-per-second "%.0f")}))))

(defn- git-sha
  "HEAD's short sha, marked -dirty when the working tree has changes."
  []
  (str (str/trim (:out (sh/sh "git" "rev-parse" "--short" "HEAD")))
       (when (seq (str/trim (:out (sh/sh "git" "status" "--porcelain" "--untracked-files=no")))) "-dirty")))

(defn run
  "Run `posteriors` (default all) and return the result rows."
  ([] (run (sort (keys models))))
  ([posteriors]
   (vec (mapcat (fn [p] (println "running" p) (flush) (run-posterior p nil)) posteriors))))

(defn record!
  "Run `posteriors` (default all), write the results and print them next to
  the previous run's; true when everything passes."
  [posteriors]
  (let [previous (previous-results)
        rows (run (or (seq posteriors) (sort (keys models))))
        stamp (.format (java.text.SimpleDateFormat. "yyyy-MM-dd-HHmmss") (java.util.Date.))
        out (first (remove #(.exists %) (map #(io/file results-dir (str stamp "-" (git-sha) (when (pos? %) (str "-" %)) ".edn"))
                                             (range))))]
    (io/make-parents out)
    (spit out (with-out-str (pp/pprint {:commit (git-sha) :posteriordb commit :date stamp
                                        :tolerances {:mean mean-tolerance :sd sd-tolerance
                                                     :z-mcse z-tolerance :min-ess min-ess}
                                        :results rows})))
    (report rows previous)
    (println "results:" (str out))
    (every? :pass? rows)))

(defn control!
  "The negative control: eight schools with τ's log-Jacobian dropped, a
  classic transform bug that leaves the posterior improper towards τ = 0.
  Every method must run and fail the comparison; true when they all do."
  []
  (let [p "eight_schools-eight_schools_noncentered"
        own (:log-density (models p))
        rows (run-posterior p (fn [^doubles th inputs] (- (double (own th inputs)) (aget th 9))))]
    (report rows nil)
    ;; detected by the comparison, not by a run that failed for another
    ;; reason; a run too short to be confident (LOW-ESS) is not passed
    ;; either, which is what the control asks
    (every? #(and (not (:error %)) (not (:pass? %))) rows)))

(defn -main [& posteriors]
  (let [ok? (if (= ["control"] posteriors) (control!) (record! posteriors))]
    (shutdown-agents)
    (System/exit (if ok? 0 1))))
