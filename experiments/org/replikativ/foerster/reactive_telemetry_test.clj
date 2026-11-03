(ns org.replikativ.foerster.reactive-telemetry-test
  "Numerical and reactive admission checks for the reviewed telemetry fixture.
  Run explicitly with experiments and test on the classpath."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.reactive-telemetry :as telemetry]
            [org.replikativ.foerster.inference.scmc :as scmc]))

(def receipt
  (delay
    (let [output "/tmp/foerster-reactive-telemetry-validated.edn"]
      (telemetry/-main output)
      (edn/read-string (slurp output)))))

(defn accurate? [errors]
  (and (every? #(< % 0.25) (:mean-in-sd errors))
       (every? #(< % 0.1) (:relative-sd errors))
       (< (:risk-absolute errors) 0.01)))

(deftest quadrature-is-resolved-and-the-posterior-is-nontrivial
  (let [r @receipt]
    (doseq [error (:grid-check r)]
      (is (every? #(< % 0.005) (:mean-in-sd error)))
      (is (every? #(< % 0.005) (:relative-sd error))))
    (doseq [ref (:references r)] (is (< (:boundary-mass ref) 1e-7)))
    ;; The covariate shift changes both identification and posterior geometry.
    (is (> (:correlation (first (:references r))) 0.6))
    (is (< (:correlation (nth (:references r) 2)) -0.6))
    (is (> (/ (get-in r [:references 0 :sd 1]) (get-in r [:references 2 :sd 1])) 4.0))))

(deftest accuracy-and-cost-are-checked-independently-of-ess
  (let [fits (:fits @receipt)
        by-key (into {} (map #(vector [(:memory %) (:seed %)] %) fits))]
    (doseq [seed [31 32 33] memory [128 300]]
      (let [fit (by-key [memory seed])]
        (is (accurate? (:errors fit)))
        (is (<= (:coreset-size fit) memory))
        (is (<= (:max-retained-coreset-size fit) memory))
        (is (<= (:max-transient-coreset-size fit) (+ memory telemetry/batch-size)))
        (is (= (:max-coreset-size fit) (:max-transient-coreset-size fit)))))
    (doseq [seed [31 32 33]]
      (is (< (:calls (by-key [128 seed])) (* 0.65 (:calls (by-key [300 seed]))))))
    (let [poor (filter #(= 8 (:memory %)) (:fits @receipt))]
      (doseq [fit poor]
        (is (not (accurate? (:errors fit))))
        ;; Quadrature errors with the NNLS refit: 0.72, 0.39, 0.75 SD (0.8
        ;; asserted, 1.03-1.12 measured, before the refit).
        (is (> (apply max (get-in fit [:errors :mean-in-sd])) 0.35))
        (is (> (get-in fit [:last-update :ess-fraction]) 0.7))))))

(deftest revisions-corrections-and-speculation
  (let [r (:reactive @receipt)]
    (is (= [:append :append :append :replay] (mapv :operation (:updates r))))
    (is (= [1 2 3 4] (mapv :revision (:updates r))))
    (doseq [update (:updates r)]
      (is (accurate? (:errors update)))
      (is (= :not-assessed (:posterior-accuracy update)))
      (is (<= (:coreset-size update) 128)))
    (is (= [:stale-base :scorer-changed :invalid-observations] (mapv :reason (:refusals r))))
    (is (:unchanged-after-refusals? r))
    (is (:duplicate-is-noop? r))
    (is (:fork-isolated? r))
    (is (= 5 (get-in r [:branch-result :revision])))
    (is (pos? (get-in r [:updates 3 :batch-splits])))
    ;; the correction starts at window 60, a snapshot: 240 windows replayed
    (is (= 60 (get-in r [:updates 3 :replayed-from])))
    (is (= 240 (get-in r [:updates 3 :replayed-observations])))
    (is (= :needs-recovery (get-in r [:stress-result :status])))
    (is (= :low-ess (get-in r [:stress-result :reason])))
    (is (= 4 (get-in r [:stress-result :revision])))
    (is (true? (get-in r [:stress-result :unchanged?])))
    (is (= :approximate-population (get-in r [:updates 3 :inference-contract :semantics])))))

(deftest selected-budget-on-independent-simulated-streams
  (doseq [seed [20261031 20261032]]
    (let [rows (telemetry/observations seed)
          ref (telemetry/reference rows 0.05)
          fit (telemetry/fit rows {:particles 800 :memory 128 :seed 31})]
      (is (accurate? (telemetry/errors (:summary fit) ref))))))

;; Historical corrections from snapshots: a plain atom stands in for the
;; world-local runtime atom, so these checks need no reactive context.

(def ^:private small {:particles 100 :memory 32 :seed 7})

(defn- full-replay
  "The corrected log replayed from the prior, as the fixture did before it
  kept snapshots."
  [rows]
  (reduce (fn [engine batch] (:engine (telemetry/advance-safely engine batch)))
          (scmc/start telemetry/model small)
          (partition-all telemetry/batch-size rows)))

(defn- publish! [owner revision rows]
  (telemetry/update-owner! owner {:revision revision :base-revision (:revision @owner)
                                  :rows rows :scorer telemetry/scorer-id}))

(defn- streamed
  "An owner after appending 180 windows, then the remaining 120."
  []
  (let [rows (telemetry/observations)
        owner (atom (telemetry/owner-state small))]
    (publish! owner 1 (subvec rows 0 180))
    (publish! owner 2 rows)
    owner))

(defn- bump
  "Rows with three more failures in the windows from `from` to `to`."
  [rows from to]
  (reduce (fn [out i] (update-in out [i :failures] #(min 8 (+ % 3)))) rows (range from to)))

(deftest a-correction-replays-from-a-snapshot-to-the-full-replay
  (let [owner (streamed)
        rows (:rows @owner)]
    (is (= [0 60 120 180 240 300] (mapv :index (:snapshots @owner))))
    (doseq [[label corrected from] [["the fixture's correction, which splits batches" (telemetry/corrected rows) 60]
                                    ["a correction after the last but one snapshot" (bump rows 250 255) 240]]]
      (testing label
        (let [o (atom @owner)
              receipt (publish! o 3 corrected)
              full (full-replay corrected)]
          (is (= :replay (:operation receipt)))
          (is (= from (:replayed-from receipt)))
          (is (true? (:compressed? full)))
          (is (< 32 from) "the core-set was compressed before the snapshot")
          (is (= full (:engine @o)))
          (is (= (mapv :index (:snapshots @owner)) (mapv :index (:snapshots @o)))))))
    (testing "low-ESS splits keep the batch boundaries and the snapshots"
      (is (pos? (:batch-splits (publish! (atom @owner) 3 (telemetry/corrected rows))))))))

(deftest replay-cost-grows-with-the-suffix
  (let [owner (streamed)
        rows (:rows @owner)
        costs (into {}
                    (for [at [10 70 250]]
                      (let [o (atom @owner)
                            corrected (bump rows at (inc at))
                            receipt (publish! o 3 corrected)
                            {:keys [index engine]} (last (filter #(<= (:index %) at) (:snapshots @owner)))
                            calls (volatile! 0)]
                        ;; the suffix replayed from the snapshot by itself
                        (binding [telemetry/*calls* calls]
                          (reduce (fn [e b] (:engine (telemetry/advance-safely e b)))
                                  engine (partition-all telemetry/batch-size (subvec corrected index))))
                        (is (= index (:replayed-from receipt)))
                        (is (= @calls (:calls receipt)))
                        [at (:calls receipt)])))]
    (is (> (costs 10) (costs 70) (costs 250)))
    (testing "a correction in the first windows replays everything"
      (let [calls (volatile! 0)]
        (binding [telemetry/*calls* calls] (full-replay (bump rows 10 11)))
        (is (= @calls (costs 10)))))))

(deftest appends-off-the-batch-grid-take-no-snapshots
  ;; Two single-window appends: the second starts inside a batch, so the
  ;; state depends on the append history and is not the replay from the
  ;; prior; no snapshot is taken from it. A correction still replays from
  ;; the last snapshot to the full replay.
  (let [owner (streamed)
        rows (:rows @owner)
        more (conj rows {:id 300 :load 1.0 :requests 8 :failures 4} {:id 301 :load 0.9 :requests 8 :failures 3})]
    (publish! owner 3 (subvec more 0 301))
    (is (:canonical? @owner))
    (publish! owner 4 more)
    (is (false? (:canonical? @owner)))
    (is (not= (full-replay more) (:engine @owner)))
    (is (= [0 60 120 180 240 300] (mapv :index (:snapshots @owner))))
    (let [corrected (bump more 280 281)
          receipt (publish! owner 5 corrected)]
      (is (= 240 (:replayed-from receipt)))
      (is (:canonical? @owner))
      (is (= (full-replay corrected) (:engine @owner))))))

(deftest snapshots-are-bounded
  (let [rows (telemetry/observations)
        owner (atom (telemetry/owner-state small))]
    (doseq [[revision end] (map vector (iterate inc 1) (range 60 301 60))]
      (publish! owner revision (subvec rows 0 end)))
    (with-redefs [telemetry/max-snapshots 2]
      (publish! owner 6 (conj rows {:id 300 :load 0.0 :requests 8 :failures 1})))
    (is (= [0 240 300] (mapv :index (:snapshots @owner))))))
