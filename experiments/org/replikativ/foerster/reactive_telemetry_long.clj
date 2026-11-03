(ns org.replikativ.foerster.reactive-telemetry-long
  "Bounded per-update cost on a long telemetry stream (experiment E1 of the
  SCMC review): the fixture's binomial-logistic model on a stream whose load
  is continuous, so no finite sufficient statistic over loads exists, with
  exact quadrature of the 2-d posterior at checkpoints.

  Arms run on the same SCMC engine (`scmc/append-batch`, i.e.
  `population/theta-smc`) with the same particles and seeds:

    [:full]            resample-move against the whole history (IBIS):
                       SCMC whose memory holds every window
    [:scmc M opts]     SCMC with memory M (opts: :refit, :projection-inflation)
    [:reservoir M]     a uniform reservoir of M past windows (Vitter's
                       Algorithm R), each weighted seen/M: the UAI paper's
                       baseline

  Per update it records likelihood evaluations by phase (reweight, move,
  projection), wall milliseconds, retained factors, particles and the ESS
  before resampling; at each checkpoint the error against quadrature
  (|Δmean|/SD per parameter, SD ratio, predicted high-load risk).

  Run `clojure -M:telemetry-long OUTPUT [option value …]` (options as EDN,
  e.g. `:windows 300 :particles 100`); the defaults are the full E1
  configuration (6000 windows, 2 data seeds, 3 inference seeds, 800
  particles), which takes about an hour and a half."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.reactive-telemetry :as telemetry]
            [org.replikativ.foerster.inference.scmc :as scmc]))

(def defaults
  {:windows 6000 :batch-size 15 :requests 8
   ;; fractions of the stream at low, mid and high load coverage, then an
   ;; optional return to low load (probes forgetting)
   :coverage [0.2 0.4 0.4] :return-to-low nil
   :data-seeds [7 8] :seeds [1 2 3] :particles 800 :checkpoints 20 :grid 121
   :arms [[:full] [:scmc 32] [:scmc 128] [:scmc 512]
          [:reservoir 32] [:reservoir 128] [:reservoir 512]]})

;; =============================================================================
;; Stream and reference
;; =============================================================================

(defn observations
  "`windows` telemetry windows with load uniform on [lo, lo + 1): lo = -1,
  -0.5 and 0 over the coverage fractions, then -1 again for the
  `return-to-low` fraction (appended after the stream) if any; failures
  are binomial at logistic(-2 + 2·load)."
  [{:keys [windows coverage return-to-low requests]} seed]
  (let [rng (java.util.Random. seed)
        [low mid] coverage
        extra (long (Math/round (* windows (or return-to-low 0.0))))]
    (mapv (fn [i]
            (let [lo (cond (< i (* low windows)) -1.0
                           (< i (* (+ low mid) windows)) -0.5
                           (< i windows) 0.0
                           :else -1.0)
                  load (+ lo (.nextDouble rng))
                  p (telemetry/sigmoid (+ -2.0 (* 2.0 load)))]
              {:id i :load load :requests requests
               :failures (count (filter #(< % p) (repeatedly requests #(.nextDouble rng))))}))
          (range (+ windows extra)))))

(defn- grid-posterior
  "Rectangular quadrature of prior × likelihood of `rows` on a k×k grid of
  `center` ± 8·`scale`."
  [rows [ca cb] [sa sb] k]
  (let [xs (double-array (map :load rows))
        ns (double-array (map :requests rows))
        ks (double-array (map :failures rows))
        m (alength xs)
        axis (fn [c s] (mapv #(+ c (* s (- (* 16.0 (/ % (dec k))) 8.0))) (range k)))
        as (axis ca sa) bs (axis cb sb)
        points (vec (for [a as b bs]
                      [a b (+ (telemetry/log-prior [a b])
                              (loop [i 0 s 0.0]
                                (if (< i m)
                                  (let [eta (+ a (* b (aget xs i)))]
                                    (recur (inc i) (+ s (- (* (aget ks i) eta) (* (aget ns i) (telemetry/softplus eta))))))
                                  s)))]))
        top (reduce max (map #(nth % 2) points))
        weighted (mapv (fn [[a b score]] [a b (Math/exp (- score top))]) points)
        z (reduce + (map #(nth % 2) weighted))
        avg (fn [f] (/ (reduce + (map (fn [[a b w]] (* w (f a b))) weighted)) z))
        ma (avg (fn [a _] a)) mb (avg (fn [_ b] b))
        va (avg (fn [a _] (* (- a ma) (- a ma))))
        vb (avg (fn [_ b] (* (- b mb) (- b mb))))
        edge? (fn [a b] (or (= a (first as)) (= a (peek as)) (= b (first bs)) (= b (peek bs))))]
    {:mean [ma mb] :sd [(Math/sqrt va) (Math/sqrt vb)]
     :correlation (/ (avg (fn [a b] (* (- a ma) (- b mb)))) (Math/sqrt (* va vb)))
     :risk-at-high-load (avg (fn [a b] (telemetry/sigmoid (+ a b))))
     :boundary-mass (/ (reduce + (map (fn [[a b w]] (if (edge? a b) w 0.0)) weighted)) z)}))

(defn reference
  "The exact posterior summary of `rows` by quadrature, recentred twice on
  its own mean ± 8 SD, starting from `guess` ({:mean :sd})."
  [rows guess k]
  (reduce (fn [r _] (grid-posterior rows (:mean r) (:sd r) k)) guess (range 3)))

;; =============================================================================
;; Arms
;; =============================================================================

(defn- reservoir-step
  "One reservoir step: SCMC's reweight, resample and move against the
  reservoir weighted seen/|R| plus the batch, then Algorithm R on the batch."
  [{:keys [reservoir seen] :as state} batch]
  (let [weight (if (seq reservoir) (/ (double seen) (count reservoir)) 0.0)
        moved (scmc/append-batch (-> state
                                     (assoc :memory Long/MAX_VALUE
                                            :coreset (mapv #(vector weight %) reservoir))
                                     (dissoc :coreset-log-likelihoods))
                                 batch)
        capacity (:reservoir-size state)
        [reservoir seen] (random/with-stream* (:seed state) [::reservoir (:steps state)]
                           #(reduce (fn [[r s] x]
                                      (let [s (inc s)]
                                        (if (< (count r) capacity)
                                          [(conj r x) s]
                                          (let [j (long (* s (d/draw (d/uniform 0.0 1.0))))]
                                            [(if (< j capacity) (assoc r j x) r) s]))))
                                    [reservoir seen] batch))]
    (-> moved
        (dissoc :coreset-log-likelihoods)
        (assoc :memory (:memory state) :reservoir reservoir :seen seen
               :coreset (mapv #(vector (/ (double seen) (count reservoir)) %) reservoir)))))

(defn- arm-start [[kind m opts] windows particles seed]
  (case kind
    :full (scmc/start telemetry/model {:particles particles :memory (max 1 windows) :seed seed})
    :scmc (scmc/start telemetry/model (merge {:particles particles :memory m :seed seed} opts))
    :reservoir (assoc (scmc/start telemetry/model {:particles particles :memory m :seed seed})
                      :reservoir [] :seen 0 :reservoir-size m)))

(defn- arm-step [[kind] state batch]
  (if (= :reservoir kind) (reservoir-step state batch) (scmc/append-batch state batch)))

(defn run-arm
  "Stream `rows` through `arm`; per update and at the `checkpoints` (window
  counts → reference summaries) record cost and error."
  [arm rows {:keys [batch-size particles]} seed checkpoints]
  (loop [state (arm-start arm (count rows) particles seed)
         batches (partition-all batch-size rows) t 0 updates [] errors []]
    (if-let [batch (first batches)]
      (let [calls (volatile! 0)
            started (System/nanoTime)
            next (binding [telemetry/*calls* calls] (arm-step arm state batch))
            ms (/ (- (System/nanoTime) started) 1e6)
            t (+ t (count batch))
            {:keys [likelihood-evaluations ess-fraction]} (:last-update next)]
        (recur next (rest batches) t
               (conj updates {:t t :calls @calls :evaluations likelihood-evaluations :ms ms
                              :retained (count (:coreset next)) :particles (count (:particles next))
                              :ess-fraction ess-fraction})
               (if-let [ref (checkpoints t)]
                 (let [fit (telemetry/particle-summary (:particles next))]
                   (conj errors {:t t :errors (telemetry/errors fit ref)
                                 :sd-ratio (mapv / (:sd fit) (:sd ref))
                                 :risk [(:risk-at-high-load fit) (:risk-at-high-load ref)]}))
                 errors)))
      {:arm arm :seed seed :updates updates :checkpoints errors
       :total-calls (reduce + (map :calls updates))
       :total-ms (reduce + (map :ms updates))
       :worst-error (reduce max 0.0 (mapcat #(get-in % [:errors :mean-in-sd]) errors))
       :final-error (reduce max 0.0 (get-in (peek errors) [:errors :mean-in-sd]))
       :moves (:moves state) :accepted (:accepted state)})))

(defn checkpoint-windows
  "`n` evenly spaced window counts on batch boundaries, ending at `total`."
  [total batch-size n]
  (let [batches (long (Math/ceil (/ total (double batch-size))))]
    (vec (distinct (map (fn [i] (min total (* batch-size (long (Math/ceil (/ (* i batches) (double n)))))))
                        (range 1 (inc n)))))))

(defn experiment
  "Run every arm on every data seed and inference seed of `config` (merged
  over `defaults`)."
  [config]
  (let [{:keys [data-seeds seeds arms batch-size checkpoints grid] :as config} (merge defaults config)]
    {:config (dissoc config :arms) :arms arms
     :runs (vec (for [data-seed data-seeds
                      :let [rows (observations config data-seed)
                            ts (checkpoint-windows (count rows) batch-size checkpoints)
                            refs (first (reduce (fn [[out guess] t]
                                                  (let [r (reference (subvec rows 0 t) guess grid)]
                                                    [(assoc out t r) r]))
                                                [{} {:mean [-2.0 0.0] :sd [1.5 1.5]}] ts))]
                      arm arms
                      seed seeds]
                  (assoc (run-arm arm rows config seed refs)
                         :data-seed data-seed
                         :references (into (sorted-map) refs))))}))

(defn- summary-table [{:keys [runs]}]
  (str "arm\tdata_seed\tseed\ttotal_calls\tlast_update_calls\tmax_retained\tworst_error_sd\tfinal_error_sd\ttotal_ms\n"
       (apply str (for [{:keys [arm data-seed seed total-calls updates worst-error final-error total-ms]} runs]
                    (str (str/join "\t" [(pr-str arm) data-seed seed total-calls (:calls (peek updates))
                                         (reduce max 0 (map :retained updates)) worst-error final-error
                                         (Math/round (double total-ms))])
                         "\n")))))

(defn -main [output & options]
  (let [config (into {} (map (fn [[k v]] [(edn/read-string k) (edn/read-string v)]) (partition 2 options)))
        result (experiment config)
        table (summary-table result)]
    (spit output (pr-str result))
    (spit (str output ".tsv") table)
    (println table)
    (shutdown-agents)))
