(ns org.replikativ.foerster.amortized-scaling
  "Loop-generated linear Gaussian state-space qualification. CPU only.
  Exact Kalman/RTS marginal references; held-out rank/coverage and predictive
  checks are separate from denoising validation and never stop training."
  (:refer-clojure :exclude [await])
  (:require [clojure.java.io :as io]
            [org.replikativ.foerster.amortized :as a]
            [org.replikativ.foerster.amortized.compiler :as c]
            [org.replikativ.foerster-raster.amortized :as gsdm]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]))

(defn simulator [n]
  (spin
   (loop [i 0 previous 0.0 out {}]
     (if (= i n) out
         (let [x (sample (d/normal (* 0.8 previous) 0.6) :id [:x i])
               y (sample (d/normal x 0.5) :id [:y i])]
           (recur (inc i) x (assoc out [:x i] x [:y i] y)))))))

(defn reference
  "Kalman filtering followed by RTS smoothing; missing y entries are allowed."
  [n evidence]
  (let [filtered (loop [i 0 mu 0.0 variance 0.0 out []]
                   (if (= i n) out
                       (let [pm (* 0.8 mu) pv (+ (* 0.64 variance) 0.36)
                             k (if (contains? evidence [:y i]) (/ pv (+ pv 0.25)) 0.0)
                             m (+ pm (* k (- (get evidence [:y i] pm) pm)))
                             v (* (- 1.0 k) pv)]
                         (recur (inc i) m v (conj out {:mean m :variance v :pred-mean pm :pred-var pv})))))]
    (loop [i (- n 2) out (mapv #(select-keys % [:mean :variance]) filtered)]
      (if (neg? i) out
          (let [f (filtered i) next-f (filtered (inc i)) next-s (out (inc i))
                gain (/ (* 0.8 (:variance f)) (:pred-var next-f))]
            (recur (dec i) (assoc out i
                                  {:mean (+ (:mean f) (* gain (- (:mean next-s) (:pred-mean next-f))))
                                   :variance (+ (:variance f) (* gain gain (- (:variance next-s) (:pred-var next-f))))})))))))

(defn- moments [xs]
  (let [mu (/ (reduce + xs) (count xs))]
    {:mean mu :variance (/ (reduce + (map #(let [e (- % mu)] (* e e)) xs)) (count xs))}))

(defn- covered? [xs truth]
  (let [xs (vec (sort xs)) n (count xs)]
    (<= (xs (int (* 0.05 n))) truth (xs (min (dec n) (int (* 0.95 n)))))))

(defn checks [fitted n masks {:keys [cases draws seed] :or {cases 16 draws 128 seed 8301}}]
  (spin
   (let [rows (await (a/examples (:task fitted) cases {:seed seed}))]
     (loop [patterns (map-indexed vector masks) reports []]
       (if (seq patterns)
         (let [[pattern mask] (first patterns)
               records (loop [i 0 records []]
                         (if (= i cases) records
                             (let [row (rows i) evidence (select-keys row mask)
                                   exact (reference n evidence)
                                   posterior (await (a/infer fitted evidence {:draws draws :seed (+ seed 1000 i)}))
                                   samples (mapv (comp m/get-value first) (m/get-particles posterior))
                                   rng (java.util.Random. (+ seed 100000 i))
                                   stats (mapv (fn [j]
                                                 (let [xs (mapv #(get % [:x j]) samples)
                                                       learned (moments xs) truth (get row [:x j]) ref (exact j)
                                                       ;; Independent future sensor noise, retaining each
                                                       ;; posterior state draw. The held-out replicate is
                                                       ;; generated from the true state and independent noise.
                                                       replicated (mapv #(+ % (* 0.5 (.nextGaussian rng))) xs)
                                                       predicted (moments replicated)
                                                       replicate-truth (+ truth (* 0.5 (.nextGaussian rng)))]
                                                   {:mean-error (- (:mean learned) (:mean ref))
                                                    :variance-ratio (/ (:variance learned) (:variance ref))
                                                    :covered? (covered? xs truth)
                                                    :rank (/ (count (filter #(< % truth) xs)) (double draws))
                                                    :predictive-mean-error (- (:mean predicted) (:mean ref))
                                                    :predictive-variance-ratio (/ (:variance predicted) (+ (:variance ref) 0.25))
                                                    :predictive-covered? (covered? replicated replicate-truth)})) (range n))]
                               (recur (inc i) (into records stats)))))]
           (let [total (count records)
                 average (fn [f] (/ (reduce + (map f records)) total))]
             (recur (rest patterns)
                    (conj reports {:pattern pattern :observed-count (count mask) :cases cases :draws draws
                                   :state-marginals total
                                   :mean-rmse (Math/sqrt (average #(let [e (:mean-error %)] (* e e))))
                                   :mean-variance-ratio (average :variance-ratio)
                                   :coverage-90 (/ (count (filter :covered? records)) (double total))
                                   :rank-histogram (frequencies (map #(min 9 (int (* 10 (:rank %)))) records))
                                   :predictive-mean-rmse (Math/sqrt (average #(let [e (:predictive-mean-error %)] (* e e))))
                                   :predictive-mean-variance-ratio (average :predictive-variance-ratio)
                                   :predictive-coverage-90 (/ (count (filter :predictive-covered? records)) (double total))}))))
         reports)))))

(defn refresh-checks!
  "Re-run posterior and fresh-replicate checks from recorded checkpoints."
  [output]
  (binding [engine/*execution-context* (context/create-execution-context)]
    (let [study (clojure.edn/read-string (slurp output))]
      (loop [remaining (:reports study) reports []]
        (if-let [report (first remaining)]
          (let [n (:n report)
                masks [(set (map #(vector :y %) (range n)))
                       (set (map #(vector :y %) (range 0 n 2)))]
                task (c/compile #'simulator {:args [n] :targets (mapv #(vector :x %) (range n))
                                             :conditionings masks :id (keyword "scaling" (str "states-" n))})
                path (str output ".states-" n ".checkpoint.edn")
                checkpoint (clojure.edn/read-string (slurp path))
                fitted (a/thaw task (gsdm/backend) checkpoint)
                checked @(checks fitted n masks {})
                reports (conj reports (assoc report :checks checked))]
            (spit path (pr-str (a/freeze (assoc fitted :validation checked))))
            (prn {:refreshed n :checks checked})
            (recur (rest remaining) reports))
          (spit output (pr-str (assoc study :reports reports
                                      :check-protocol {:cases 16 :draws 128 :seed 8301
                                                       :reference :kalman-rts
                                                       :predictive :independent-replicate-observations}))))))))

(defn run-study! [output updates]
  (let [output (or output "experiments/results/amortized/scaling.edn")
        updates (or updates 30000)]
    (binding [engine/*execution-context* (context/create-execution-context)]
      (loop [sizes [2 8 32] reports []]
        (when-let [n (first sizes)]
          (let [all-y (set (map #(vector :y %) (range n)))
                partial-y (set (map #(vector :y %) (range 0 n 2)))
                masks [all-y partial-y]
                task (c/compile #'simulator {:args [n] :targets (mapv #(vector :x %) (range n))
                                             :conditionings masks :id (keyword "scaling" (str "states-" n))})
                fitted @(a/train task (gsdm/backend {:emb-dim 8 :n-layers 2})
                                 {:simulations 2048 :validation-simulations 128 :updates updates
                                  :validation-every 1000 :validation-cases 128 :rate 0.0001 :seed 7301
                                  :on-progress #(prn (assoc (select-keys % [:status :updates :validation-loss :seconds]) :n n))})
                report {:n n :dimension (:dimension task) :source-edges (count (:attention-edges task))
                        :compilation (:compilation task)
                        :backend (dissoc (:backend-info fitted) :training)
                        :training (get-in fitted [:payload :training])
                        :checks @(checks fitted n masks {})}
                reports (conj reports report)]
            (io/make-parents output)
            (spit output (pr-str {:scope {:backend :cpu-jvm-aot :dtype :double :calibration :exploratory
                                          :note "Correlated state marginals; coverage counts are not independent trials."}
                                  :reports reports}))
            (spit (str output ".states-" n ".checkpoint.edn")
                  (pr-str (a/freeze (assoc fitted :validation (:checks report)))))
            (prn (dissoc report :training :compilation))
            (recur (rest sizes) reports)))))
    nil))

(defn -main [& [output updates-text]]
  (run-study! output (when updates-text (Long/parseLong updates-text)))
  (shutdown-agents))
