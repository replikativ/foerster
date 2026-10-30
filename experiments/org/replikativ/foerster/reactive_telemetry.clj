(ns org.replikativ.foerster.reactive-telemetry
  "A reviewed fixed-shape telemetry fixture; not automatic update admission."
  (:require [clojure.string :as str]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.scmc :as scmc]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.impl.simple :as simple]
            [org.replikativ.spindel.effects.track :refer [track]]
            [org.replikativ.spindel.signal :as sig]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:dynamic *calls* nil)
(def scorer-id :binomial-logistic-v1)
(def batch-size 15)
(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))
(defn softplus [x] (+ (max x 0.0) (Math/log1p (Math/exp (- (Math/abs (double x)))))))
(defn log-choose [n k]
  (reduce + 0.0 (map #(Math/log (/ (double (- (inc n) %)) %)) (range 1 (inc k)))))
(defn log-lik [[a b] {:keys [load requests failures]}]
  (when *calls* (vswap! *calls* inc))
  (let [eta (+ a (* b load))]
    (+ (log-choose requests failures) (* failures eta) (- (* requests (softplus eta))))))
(defn log-prior [[a b]]
  (+ (d/logpdf (d/normal -2.0 1.5) a) (d/logpdf (d/normal 0.0 1.5) b)))
(def model {:prior-sample #(vector (d/draw (d/normal -2.0 1.5)) (d/draw (d/normal 0.0 1.5)))
            :prior-log-density log-prior :log-lik log-lik})

(defn observations
  ([] (observations 20261030))
  ([seed]
   (let [rng (java.util.Random. seed)]
     (mapv (fn [i]
             (let [load (/ (+ (cond (< i 60) -10 (< i 180) -5 :else 0) (.nextInt rng 11)) 10.0)
                   p (sigmoid (+ -2.0 (* 2.0 load)))]
               {:id i :load load :requests 8
                :failures (count (filter #(< % p) (repeatedly 8 #(.nextDouble rng))))}))
           (range 300)))))

(defn particle-summary [qs]
  (let [n (count qs)
        a (/ (reduce + (map first qs)) n) b (/ (reduce + (map second qs)) n)
        va (/ (reduce + (map #(let [x (- (first %) a)] (* x x)) qs)) n)
        vb (/ (reduce + (map #(let [x (- (second %) b)] (* x x)) qs)) n)
        cov (/ (reduce + (map #(* (- (first %) a) (- (second %) b)) qs)) n)]
    {:mean [a b] :sd [(Math/sqrt va) (Math/sqrt vb)]
     :correlation (/ cov (Math/sqrt (* va vb)))
     :risk-at-high-load (/ (reduce + (map #(sigmoid (+ (first %) (second %))) qs)) n)}))

(defn reference
  "Independent rectangular quadrature, using binomial sufficient statistics
  at each of the 21 load values. Constants cancel in normalized posterior."
  [rows spacing]
  (let [groups (mapv (fn [[x rs]] [x (reduce + (map :requests rs)) (reduce + (map :failures rs))])
                     (group-by :load rows))
        axes (mapv (fn [[lo hi]] (mapv #(+ lo (* spacing %))
                                       (range (inc (long (Math/round (/ (- hi lo) spacing)))))))
                   [[-6.0 1.0] [-3.0 6.0]])
        points (vec (for [a (axes 0) b (axes 1)]
                      (let [score (+ (log-prior [a b])
                                     (reduce + (map (fn [[x n k]]
                                                      (let [eta (+ a (* b x))]
                                                        (- (* k eta) (* n (softplus eta))))) groups)))]
                        [a b score])))
        top (reduce max (map #(nth % 2) points))
        weighted (mapv (fn [[a b score]] [a b (Math/exp (- score top))]) points)
        z (reduce + (map #(nth % 2) weighted))
        avg (fn [f] (/ (reduce + (map (fn [[a b w]] (* w (f a b))) weighted)) z))
        ma (avg (fn [a _] a)) mb (avg (fn [_ b] b))
        va (avg (fn [a _] (* (- a ma) (- a ma))))
        vb (avg (fn [_ b] (* (- b mb) (- b mb))))
        cov (avg (fn [a b] (* (- a ma) (- b mb))))
        boundary (/ (reduce + (map (fn [[a b w]]
                                     (if (or (= a (first (axes 0))) (= a (peek (axes 0)))
                                             (= b (first (axes 1))) (= b (peek (axes 1)))) w 0.0)) weighted)) z)]
    {:mean [ma mb] :sd [(Math/sqrt va) (Math/sqrt vb)]
     :correlation (/ cov (Math/sqrt (* va vb)))
     :risk-at-high-load (avg (fn [a b] (sigmoid (+ a b))))
     :grid-spacing spacing :boundary-mass boundary}))

(defn errors [fit ref]
  {:mean-in-sd (mapv #(/ (Math/abs (- %1 %2)) %3) (:mean fit) (:mean ref) (:sd ref))
   :relative-sd (mapv #(Math/abs (- 1.0 (/ %1 %2))) (:sd fit) (:sd ref))
   :risk-absolute (Math/abs (- (:risk-at-high-load fit) (:risk-at-high-load ref)))})

(declare coreset-audit)

(defn fit [rows opts & [ref]]
  (let [counter (volatile! 0) started (System/nanoTime)]
    (binding [*calls* counter]
      (let [state (reduce scmc/append-batch (scmc/start model opts) (partition-all batch-size rows))]
        (cond-> {:summary (particle-summary (:particles state))
                 :calls @counter :elapsed-seconds (/ (- (System/nanoTime) started) 1e9)
                 :coreset-size (count (:coreset state)) :max-coreset-size (:max-coreset-size state)
                 :moves (:moves state) :acceptance (/ (:accepted state) (double (:moves state)))
                 :last-update (:last-update state) :contract (:inference-contract (scmc/result state))}
          ref (assoc :audit (coreset-audit state rows ref)))))))

(defn coreset-audit
  "A full-history audit on nine parameter probes; not a bounded-stream
  diagnostic and not a convergence certificate. Centering removes constants."
  [state rows ref]
  (let [[a b] (:mean ref) [sa sb] (:sd ref)
        residuals (vec (for [i [-1.0 0.0 1.0] j [-1.0 0.0 1.0]]
                         (let [q [(+ a (* i sa)) (+ b (* j sb))]]
                           (- (reduce + (map (fn [[w row]] (* w (log-lik q row))) (:coreset state)))
                              (reduce + (map #(log-lik q %) rows))))))
        mean (/ (reduce + residuals) (count residuals))]
    {:centered-log-likelihood-rms
     (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) residuals)) (count residuals)))
     :probes (count residuals) :requires-full-history? true}))

(defn valid-rows? [rows]
  (and (vector? rows) (= (count rows) (count (set (map :id rows))))
       (every? (fn [{:keys [id load requests failures]}]
                 (and (some? id) (number? load) (Double/isFinite (double load))
                      (<= -1.0 load 1.0) (pos-int? requests)
                      (nat-int? failures) (<= failures requests))) rows)))

(defn advance-safely
  "Retry a low-ESS candidate as smaller batches, from its persistent parent.
  Refuse if an individual observation remains degenerate. Adaptive scheduling
  does not certify evidence or posterior accuracy."
  [engine batch]
  (let [batch (vec batch)
        next (scmc/append-batch engine batch)
        ess (get-in next [:last-update :ess-fraction])]
    (cond
      (>= ess 0.1) {:engine next :minimum-ess ess :splits 0}
      (= 1 (count batch)) (throw (ex-info "Observation needs recovery" {:type ::low-ess :ess ess}))
      :else
      (let [middle (quot (count batch) 2)
            left (advance-safely engine (subvec batch 0 middle))
            right (advance-safely (:engine left) (subvec batch middle))]
        {:engine (:engine right) :minimum-ess (min (:minimum-ess left) (:minimum-ess right))
         :splits (+ 1 (:splits left) (:splits right))}))))

(defn update-owner!
  "Single synchronous tracked consumer only. All inference precedes reset!;
  no RNG or inference in a CAS callback. Ownership is world-local."
  [owner snapshot opts]
  (let [previous @owner
        {:keys [revision base-revision rows scorer]} snapshot
        reason (cond
                 (not= scorer scorer-id) :scorer-changed
                 (not (valid-rows? rows)) :invalid-observations
                 (= revision (:revision previous)) (when (not= rows (:rows previous)) :revision-conflict)
                 (not= base-revision (:revision previous)) :stale-base
                 (not (and (integer? revision) (> revision (:revision previous)))) :stale-revision)]
    (cond
      reason {:status :rejected :reason reason :revision (:revision previous)}
      (= revision (:revision previous)) (:receipt previous)
      :else
      (try
        (let [old (:rows previous)
              append? (and (<= (count old) (count rows)) (= old (subvec rows 0 (count old))))
              operation (if append? :append :replay)
              incoming (if append? (subvec rows (count old)) rows)
              counter (volatile! 0)
              started (System/nanoTime)
              advanced (binding [*calls* counter]
                         (reduce (fn [{:keys [engine minimum-ess splits]} batch]
                                   (let [next (advance-safely engine batch)]
                                     {:engine (:engine next) :minimum-ess (min minimum-ess (:minimum-ess next))
                                      :splits (+ splits (:splits next))}))
                                 {:engine (if append? (:engine previous) (scmc/start model opts)) :minimum-ess 1.0 :splits 0}
                                 (partition-all batch-size incoming)))
              state (:engine advanced)
              receipt {:status :updated :operation operation :revision revision
                       :observations (count rows) :calls @counter
                       :elapsed-seconds (/ (- (System/nanoTime) started) 1e9)
                       :summary (particle-summary (:particles state))
                       :minimum-ess-fraction (:minimum-ess advanced)
                       :batch-splits (:splits advanced)
                       :weight-health :adequate
                       :posterior-accuracy :not-assessed
                       :coreset-size (count (:coreset state)) :last-update (:last-update state)
                       :inference-contract (:inference-contract (scmc/result state))}]
          (reset! owner {:revision revision :rows rows :engine state :receipt receipt})
          receipt)
        (catch clojure.lang.ExceptionInfo e
          (if (= ::low-ess (:type (ex-data e)))
            {:status :needs-recovery :reason :low-ess :revision (:revision previous)
             :requested-revision revision :ess-fraction (:ess (ex-data e))}
            (throw e)))))))

(defn corrected [rows]
  ;; Repair a historical ingest bug that undercounted ten older windows.
  (reduce (fn [out i] (update-in out [i :failures] #(min 8 (+ % 3)))) rows (range 60 70)))

(defn reactive-demo [opts]
  (let [context (ctx/create-execution-context)]
    (try
      (binding [ec/*execution-context* context]
        (let [rows (observations)
              source (sig/signal {:revision 0 :base-revision 0 :rows [] :scorer scorer-id})
              owner (sa/atom {:revision 0 :rows [] :engine (scmc/start model opts)
                              :receipt {:status :ready :revision 0}})
              consumer (spin (let [{:keys [new]} (track source)] (update-owner! owner new opts)))
              _ @consumer
              publish (fn [snapshot]
                        (reset! source snapshot)
                        (when-not (simple/await-drain-complete! ec/*execution-context* :timeout-ms 30000)
                          (throw (ex-info "Reactive telemetry timed out" {})))
                        @consumer)
              snapshots [{:revision 1 :base-revision 0 :rows (subvec rows 0 60)}
                         {:revision 2 :base-revision 1 :rows (subvec rows 0 180)}
                         {:revision 3 :base-revision 2 :rows rows}
                         {:revision 4 :base-revision 3 :rows (corrected rows)}]
              updates (mapv #(publish (assoc % :scorer scorer-id)) snapshots)
              before (:engine @owner)
              refusals [(publish {:revision 5 :base-revision 2 :rows rows :scorer scorer-id})
                        (publish {:revision 5 :base-revision 4 :rows rows :scorer :other})
                        (publish {:revision 5 :base-revision 4 :rows [(first rows) (first rows)] :scorer scorer-id})]
              stable? (= before (:engine @owner))
              duplicate (publish (assoc (peek snapshots) :scorer scorer-id))
              branch (ctx/fork-context context)
              branch-result (try
                              (binding [ec/*execution-context* branch]
                                (publish {:revision 5 :base-revision 4 :scorer scorer-id
                                          :rows (conj (corrected rows) {:id 300 :load 1.0 :requests 8 :failures 8})}))
                              (finally (ctx/close-context! branch)))
              isolated? (and (= 5 (:revision branch-result)) (= 4 (:revision @owner))
                             (= before (:engine @owner)) (= duplicate @consumer))
              stress-branch (ctx/fork-context context)
              stress-result (try
                              (binding [ec/*execution-context* stress-branch]
                                (let [out (publish {:revision 5 :base-revision 4 :scorer scorer-id
                                                    :rows (conj (corrected rows)
                                                                {:id 300 :load 1.0 :requests 10000 :failures 0})})]
                                  (assoc out :unchanged? (= before (:engine @owner)))))
                              (finally (ctx/close-context! stress-branch)))]
          {:updates updates :refusals refusals :unchanged-after-refusals? stable?
           :duplicate-is-noop? (= duplicate (peek updates))
           :fork-isolated? isolated? :branch-result branch-result :stress-result stress-result}))
      (finally (ctx/close-context! context)))))

(defn write-tables! [result output]
  (spit (str output ".fits.tsv")
        (str "memory\tseed\tmean_error_sd\tsd_error_fraction\tlikelihood_calls\taudit_rms\n"
             (apply str (map (fn [fit]
                               (str (str/join "\t" [(:memory fit) (:seed fit)
                                                    (apply max (get-in fit [:errors :mean-in-sd]))
                                                    (apply max (get-in fit [:errors :relative-sd]))
                                                    (:calls fit) (get-in fit [:audit :centered-log-likelihood-rms])]) "\n"))
                             (:fits result)))))
  (spit (str output ".updates.tsv")
        (str "revision\treference_risk\tscmc_risk\n"
             (apply str (map (fn [update ref]
                               (str (str/join "\t" [(:revision update) (:risk-at-high-load ref)
                                                    (get-in update [:summary :risk-at-high-load])]) "\n"))
                             (get-in result [:reactive :updates]) (:references result))))))

(defn -main [output]
  (let [rows (observations)
        data [(subvec rows 0 60) (subvec rows 0 180) rows (corrected rows)]
        references (mapv #(reference % 0.05) data)
        demo (reactive-demo {:particles 800 :memory 128 :seed 31})
        result {:model :binomial-logistic-v1 :truth [-2.0 2.0] :batch-size batch-size
                :references references
                :grid-check (mapv #(errors (reference % 0.1) %2) data references)
                :reactive (update demo :updates (fn [updates]
                                                  (mapv #(assoc %1 :errors (errors (:summary %1) %2)) updates references)))
                :fits (vec (for [memory [8 32 64 128 300] seed [31 32 33]]
                             (let [r (fit rows {:particles 800 :memory memory :seed seed} (references 2))]
                               (assoc r :memory memory :seed seed :errors (errors (:summary r) (references 2))))))
                :heap-max-mib (/ (.maxMemory (Runtime/getRuntime)) 1048576)}]
    (spit output (pr-str result))
    (write-tables! result output)
    (dissoc result :fits)))
