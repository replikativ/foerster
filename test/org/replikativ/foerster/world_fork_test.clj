(ns org.replikativ.foerster.world-fork-test
  "`:world-policy :fork` on engine v2: inference forks the caller's world
  (once per inference for methods of several runs), every particle, branch,
  chain or iteration runs in a world of its own descending from that fork,
  sees the caller's state as inference started and its own writes, nothing
  reaches the caller, and every fork is released when inference ends."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.foerster.moves.kernel :as kernel]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.inference.smc :as smc]
            [org.replikativ.foerster.inference.smc2 :as smc2]
            [org.replikativ.foerster.runtime.async :refer [await-op]]
            [org.replikativ.foerster.runtime.canonical :as canonical]
            [org.replikativ.foerster.runtime.execution :as executor]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]))

(defn- released? [w]
  (try (v2/get-in-world w [::probe]) false
       (catch clojure.lang.ExceptionInfo e (boolean (:spindel/released (ex-data e))))))

(defmacro ^:private capturing-forks
  "Run `body` with every host fork of a world made meanwhile in `forks`."
  [[forks] & body]
  `(let [~forks (atom [])
         fork# v2/fork!]
     (with-redefs [v2/fork! (fn [& args#]
                              (let [w# (apply fork# args#)]
                                (swap! ~forks conj w#)
                                w#))]
       ~@body)))

(defn- all-released? [forks]
  (and (seq forks) (every? released? forks)))

(defmacro ^:private in-root
  "Run `body` with a fresh caller world bound, released afterwards."
  [& body]
  `(let [root# (v2/world)]
     (try
       (binding [v2/*world* root#] ~@body)
       (finally (v2/release-world! root#)))))

(defn- infer!
  "The value of the inference task `(run)`, created inside a task body of
  the caller's world (as a model calling inference would)."
  [run]
  (v2/value (task (await-op (run))) 60000 ::timed-out))

(defn- failure [run]
  (v2/value (task (try (await-op (run)) :unexpected-success
                       (catch Throwable error error)))
            60000 ::timed-out))

(def ^:private F {:world-policy :fork})

;; -----------------------------------------------------------------------------
;; Every method runs on the caller's state, in isolation
;; -----------------------------------------------------------------------------

(defn- writer
  "A model that reads the caller's `log` and appends its own entry."
  [log]
  (task
   (let [x (sample (dist/normal 0.0 1.0) :id :x)]
     (observe (dist/normal x 1.0) 0.5 :id :y)
     (swap! log conj :mine)
     @log)))

(def ^:private particle-methods
  {:smc #(inference/smc-infer %1 4 (assoc %2 :resample-threshold 2.0))
   :importance #(inference/importance-sampling %1 4 %2)
   :pimh #(inference/pimh-infer %1 3 2 %2)
   :bbvi #(inference/bbvi-infer %1 4 2 %2)
   :pgibbs #(inference/pgibbs-infer %1 4 3 %2)
   :pgas #(inference/pgas-infer %1 4 3 %2)
   :ipmcmc #(inference/ipmcmc-infer %1 3 2 (assoc %2 :num-nodes 2 :num-csmc-nodes 1))})

(deftest every-method-sees-the-caller-s-state-and-its-own-writes-only
  (doseq [[method run] particle-methods]
    (in-root
     (let [log (v2/atom [:root])]
       (capturing-forks [forks]
                        (let [posterior (infer! #(run (fn [] (writer log)) F))]
                          (is (seq (measure/get-contexts posterior)) (str method))
                          (is (every? #{[:root :mine]} (map measure/get-value (measure/get-contexts posterior)))
                              (str method ": each particle sees the caller's state and its own write"))
                          (is (= [:root] @log) (str method ": nothing reaches the caller"))
                          (is (all-released? @forks) (str method ": every fork is released"))))))))

(deftest batched-smc-runs-on-the-caller-s-state
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (in-root
       (let [log (v2/atom [:root])
             model (fn [] (task (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                  (observe (dist/normal x 1.0) 0.5 :id :y)
                                  (observe (dist/normal x 1.0) 0.7 :id :z)
                                  (swap! log conj :mine)
                                  [x @log])))
             posterior (infer! #(inference/smc-infer model 400 (assoc F :batch 16 :executor exec
                                                                    :resample-threshold 1.0 :seed 3)))
             values (mapv measure/get-value (measure/get-contexts posterior))
             ws (mapv #(Math/exp (second %)) (measure/get-particles posterior))
             mean (/ (reduce + (map * (map first values) ws)) (reduce + ws))]
         (is (= 400 (count values)))
         (is (every? #{[:root :mine]} (map second values)))
         (is (= [:root] @log))
         (is (< (Math/abs (- mean 0.4)) 0.15) (str mean " vs 0.4"))
         (is (some #(<= (:size %) 16) (:history posterior)) "resampled in batches")))
      (finally (.close ^java.lang.AutoCloseable exec)))))

(deftest every-particle-runs-the-whole-model
  ;; a model's effects may be random without a sample site: each particle
  ;; makes its own, none shares another's prefix
  (in-root
   (let [calls (atom 0)
         posterior (infer! #(inference/smc-infer (fn [] (task (let [answer (swap! calls inc)]
                                                              (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                                              answer)))
                                               4 F))]
     (is (= 4 @calls))
     (is (= #{1 2 3 4} (set (map measure/get-value (measure/get-contexts posterior))))))))

(deftest every-lineage-ends-once
  ;; a resampled world continues in its copies instead of unwinding: the
  ;; model ends once per particle, in its final world
  (let [exec (executor/thread-pool-executor {:threads 4})
        finalized (atom [])
        n 3]
    (try
      (in-root
       (let [posterior (infer! #(inference/smc-infer
                               (fn [] (task (try (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                                 :done
                                                 (finally (swap! finalized conj (v2/current-world))))))
                               n (assoc F :executor exec :resample-threshold 2.0)))]
         (is (= (repeat n :done) (map measure/get-value (measure/get-contexts posterior))))
         (is (= n (count @finalized)))
         (is (apply distinct? (map #(System/identityHashCode %) @finalized)) "each world reaches one end")))
      (finally (.close ^java.lang.AutoCloseable exec)))))

(deftest a-model-failure-fails-inference-and-releases-every-fork
  (in-root
   (capturing-forks [forks]
                    (let [result (failure #(inference/smc-infer
                                            (fn [] (task (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                                         (throw (ex-info "model failed" {:stage :model}))))
                                            3 F))]
                      (is (= ::inference/inference-failed (:type (ex-data result))))
                      (is (= {:stage :model} (ex-data (ex-cause result))))
                      (is (all-released? @forks))))))

(deftest ^:engine-v2-gap cancelling-inference-unwinds-its-particles
  ;; the public task is cancelled, but its driver (a CPS operation behind
  ;; v2/external) is not told: the particles keep running in their worlds
  ;; and the session is not closed. external-of returns no cancel fn yet.
  (doseq [policy [:fresh :fork]]
    (in-root
     (let [entered (promise)
           cleaned (promise)
           t (inference/smc-infer (fn [] (task (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                               (try (deliver entered true)
                                                    (v2/await (v2/deferred))
                                                    (finally (deliver cleaned true)))))
                                  2 {:world-policy policy})]
       (is (= true (deref entered 5000 ::timed-out)))
       (v2/cancel-task! t)
       (is (= :cancelled (try (v2/value t 5000 ::timed-out)
                              (catch Exception e (when (v2/cancelled? e) :cancelled)))))
       (is (= true (deref cleaned 5000 ::timed-out)) (str policy ": the blocked particle unwinds"))))))

;; -----------------------------------------------------------------------------
;; Methods of several runs fork the caller once
;; -----------------------------------------------------------------------------

(def ^:private sweep-methods
  {:pimh #(inference/pimh-infer %1 3 3 %2)
   :pgibbs #(inference/pgibbs-infer %1 3 3 %2)
   :pgas #(inference/pgas-infer %1 3 3 %2)
   :ipmcmc #(inference/ipmcmc-infer %1 3 2 (assoc %2 :num-nodes 2 :num-csmc-nodes 1))
   :bbvi #(inference/bbvi-infer %1 3 2 %2)})

(deftest every-run-sees-the-caller-s-state-as-inference-started
  ;; the caller's state changes after the first run: no later run sees it,
  ;; since every run forks the inference's one frozen fork of the caller
  (doseq [[method run] sweep-methods]
    (in-root
     (let [caller v2/*world*
           state (v2/atom :before)
           log (v2/atom [:root])
           runs (atom 0)
           smc (var-get #'smc/smc)]
       (with-redefs [smc/smc (fn [& args]
                               (when (= 2 (swap! runs inc))
                                 (binding [v2/*world* caller] (reset! state :changed)))
                               (apply smc args))]
         (let [posterior (infer! #(run (fn [] (task (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                                    (observe (dist/normal x 1.0) 0.5 :id :y)
                                                    (swap! log conj :mine)
                                                    [@state @log])))
                                     F))]
           (is (> @runs 1) (str method ": several runs"))
           (is (= :changed @state) (str method ": the caller changed meanwhile"))
           (is (every? #{[:before [:root :mine]]} (map measure/get-value (measure/get-contexts posterior)))
               (str method ": every run sees the state as inference started, and its own write"))
           (is (= [:root] @log) (str method))))))))

;; -----------------------------------------------------------------------------
;; A model serving the caller's queue
;; -----------------------------------------------------------------------------

(def ^:private backlog {:free-at 1.0 :jobs [{:id :in-flight :finish 1.0}]})

(def ^:private readings [{:id :a :arrival 0.5 :work 1.0 :reported 1.9}
                         {:id :b :arrival 1.2 :work 0.7 :reported 2.7}
                         {:id :c :arrival 3.0 :work 1.2 :reported 4.1}])

(defn- serve!
  "Serve `job` on the queue held by the atom `queue`; its finishing time."
  [queue {:keys [id arrival work]} speed jitter]
  (let [start (max arrival (:free-at @queue))
        finish (+ start (/ (* work jitter) speed))]
    (swap! queue (fn [state]
                   (-> state
                       (assoc :free-at finish)
                       (update :jobs conj {:id id :start start :finish finish}))))
    finish))

(defn- queue-model
  "The paper's queue: the server's speed and every job's jitter are unknown;
  completions are reported with noise. It serves on the caller's queue."
  [queue]
  (task
   (let [speed (sample (dist/categorical {0.8 0.5, 1.4 0.5}) :id :speed)]
     (doseq [{:keys [id reported] :as job} readings]
       (let [jitter (sample (dist/categorical {0.8 0.5, 1.2 0.5}) :id [:jitter id])
             finish (serve! queue job speed jitter)]
         (observe (dist/normal finish 0.35) reported :id [:completion id])))
     {:speed speed :queue @queue})))

(defn- queue-reference
  "An independent sum over the 16 speed/jitter paths: [P(speed = 1.4 | y) log Z]."
  []
  (let [paths (for [speed [0.8 1.4] a [0.8 1.2] b [0.8 1.2] c [0.8 1.2]]
                (let [finishes (rest (reductions (fn [free [job jitter]]
                                                   (+ (max free (:arrival job)) (/ (* (:work job) jitter) speed)))
                                                 (:free-at backlog)
                                                 (map vector readings [a b c])))]
                  {:speed speed
                   :p (* (/ 1.0 16) (Math/exp (reduce + (map (fn [job finish]
                                                               (dist/logpdf (dist/normal finish 0.35) (:reported job)))
                                                             readings finishes))))}))
        z (reduce + (map :p paths))]
    [(/ (reduce + (map :p (filter #(= 1.4 (:speed %)) paths))) z) (Math/log z)]))

(def ^:private p-fast (first (queue-reference)))

(defn- served-once? [values]
  (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values))

(defn- fast-fraction [posterior]
  (let [values (map measure/get-value (measure/get-contexts posterior))]
    (/ (count (filter #(= 1.4 (:speed %)) values)) (double (count values)))))

(defn- weighted-fast [posterior]
  (let [ps (measure/get-particles posterior)
        ws (mapv #(Math/exp (second %)) ps)]
    (/ (reduce + (map (fn [[s _] w] (if (= 1.4 (:speed (measure/get-value s))) w 0.0)) ps ws))
       (reduce + ws))))

(deftest enumeration-runs-on-the-caller-s-state
  (in-root
   (let [[p log-z] (queue-reference)
         queue (v2/atom backlog)]
     (capturing-forks [forks]
                      (let [posterior (infer! #(inference/infer (fn [] (queue-model queue)) (assoc F :method :enumerate)))
                            values (mapv measure/get-value (measure/get-contexts posterior))]
                        (is (< (Math/abs (- p 0.5659282411891152)) 1e-12) "the reference")
                        (is (< (Math/abs (- p (:mean (inference/query posterior #(if (= 1.4 (:speed %)) 1.0 0.0)))))
                               1e-12))
                        (is (< (Math/abs (- log-z (measure/log-marginal posterior))) 1e-12))
                        (testing "every branch served on the caller's backlog, in a queue of its own"
                          (is (= 16 (count values)))
                          (is (every? #(= (:jobs backlog) (subvec (:jobs (:queue %)) 0 1)) values))
                          (is (served-once? values))
                          (is (= 16 (count (set (map :queue values))))))
                        (is (= backlog @queue) "the caller's queue is unchanged")
                        (is (all-released? @forks)))))))

(deftest enumeration-branches-write-in-isolation
  (in-root
   (let [log (v2/atom [:root])
         posterior (infer! #(inference/infer (fn [] (task (let [x (sample (dist/flip 0.5) :id :x)]
                                                          (swap! log conj x)
                                                          (let [y (sample (dist/categorical {:a 0.2 :b 0.8}) :id :y)]
                                                            (swap! log conj y)
                                                            @log))))
                                           (assoc F :method :enumerate)))]
     (is (= #{[:root true :a] [:root true :b] [:root false :a] [:root false :b]}
            (set (map (comp measure/get-value first) (measure/get-particles posterior))))
         "each branch sees its own writes and no other branch's")
     (is (= [:root] @log))
     (is (< (Math/abs (measure/log-marginal posterior)) 1e-12)))))

(def ^:private gibbs-sweep
  (kernel/cycle 600 (mapv #(kernel/finite-gibbs-kernel 1 %) [:speed [:jitter :a] [:jitter :b] [:jitter :c]])
                {:samples :all :burn 20}))

(deftest markov-chains-run-on-the-caller-s-state
  (doseq [[method run] [[:mh #(inference/infer %1 (assoc F :method :mh :iterations 2500 :chains 4 :burn 100 :seed %2))]
                        [:gibbs #(inference/kernel-infer %1 gibbs-sweep 4 (assoc F :seed %2))]]
          seed [1 2]]
    (in-root
     (let [queue (v2/atom backlog)]
       (capturing-forks [forks]
                        (let [posterior (infer! #(run (fn [] (queue-model queue)) seed))]
                          ;; seed-to-seed SD over 12 seeds: MH 0.058, Gibbs 0.028
                          (is (< (Math/abs (- p-fast (fast-fraction posterior))) (if (= :mh method) 0.2 0.1))
                              (str method " " seed ": " (fast-fraction posterior) " vs exact " p-fast))
                          (is (served-once? (map measure/get-value (measure/get-contexts posterior)))
                              (str method ": every state served the caller's backlog, once"))
                          (is (= backlog @queue) (str method ": the caller's queue is unchanged"))
                          (is (all-released? @forks) (str method))))))))

(deftest tempered-smc-runs-on-the-caller-s-state
  (in-root
   (let [queue (v2/atom backlog)
         posterior (infer! #(inference/tempered-infer (fn [] (queue-model queue)) 300 (assoc F :seed 5)))]
     (is (< (Math/abs (- p-fast (weighted-fast posterior))) 0.08)
         (str (weighted-fast posterior) " vs exact " p-fast))
     (is (served-once? (map measure/get-value (measure/get-contexts posterior))))
     (is (= backlog @queue)))))

(deftest pmmh-runs-every-iteration-on-the-caller-s-state
  ;; θ = the server's speed, the jitters filtered by SMC: P(speed = 1.4 | y)
  (in-root
   (let [queue (v2/atom backlog)
         posterior (infer! #(inference/infer (fn [] (queue-model queue))
                                           (assoc F :method :pmmh :particles 60 :iterations 600 :burn 50
                                                  :params #{:speed}
                                                  :propose (fn [{:keys [speed]}]
                                                             {:theta {:speed (if (= speed 0.8) 1.4 0.8)}
                                                              :log-ratio 0.0})
                                                  :seed 21)))]
     (is (< (Math/abs (- p-fast (fast-fraction posterior))) 0.1)
         (str (fast-fraction posterior) " vs exact " p-fast))
     (is (served-once? (map measure/get-value (measure/get-contexts posterior))))
     (is (= backlog @queue)))))

(defn- program-run [queue n seed]
  (infer! #(ip/run {:seed seed :world-policy :fork}
                 (fn [scope]
                   (task
                    ;; importance weights, resampling (copies of the
                    ;; particles' worlds), then moves
                    (let [p (await-op (ip/initialize scope (ip/trace-target (fn [] (queue-model queue)) F) n))
                          p (ip/advance scope p (ip/temperature (:target p) 1.0))
                          p (await-op (ip/resample scope p))
                          p (await-op (ip/move scope p (kernel/single-site-mh-kernel 2)))]
                      (ip/finish scope p)))))))

(deftest inference-programs-run-trace-targets-on-the-caller-s-state
  (in-root
   (let [queue (v2/atom backlog)
         posterior (program-run queue 600 42)]
     (is (< (Math/abs (- p-fast (weighted-fast posterior))) 0.08)
         (str (weighted-fast posterior) " vs exact " p-fast))
     (is (served-once? (map measure/get-value (measure/get-contexts posterior))))
     (is (= backlog @queue)))
   (testing "resampling copies particles in their worlds"
     (let [queue (v2/atom backlog)
           posterior (program-run queue 40 7)]
       (is (= 40 (count (measure/get-particles posterior))))
       (is (served-once? (map measure/get-value (measure/get-contexts posterior))))
       (is (= backlog @queue))))
   (testing "a :fork target needs a :fork program"
     (is (thrown-with-msg? Exception #"world-policy :fork"
                           (infer! #(ip/run {:seed 1}
                                          (fn [scope]
                                            (task (ip/finish scope (await-op (ip/initialize scope (ip/trace-target (fn [] (queue-model (v2/atom backlog))) F)
                                                                                            4))))))))))))

;; -----------------------------------------------------------------------------
;; Blocks read the caller's state as inputs
;; -----------------------------------------------------------------------------

(def ^:private centred
  (block/block {:block/id :centred :block/target :complete-conditional
                :block/latents [{:name :x :shape [] :support :real}]}
               {:log-density (fn [th {:keys [center]}] (let [d (- (aget ^doubles th 0) center)] (* -0.5 d d)))
                :value+grad (fn [th {:keys [center]}]
                              (let [d (- (aget ^doubles th 0) center)]
                                [(* -0.5 d d) (double-array [(- d)])]))}))

(deftest nuts-reads-the-caller-s-state-as-block-inputs
  (in-root
   (let [center (v2/atom 3.0)
         log (v2/atom [])
         posterior (infer! #(inference/infer (fn [] (task (let [[x] (sample (block/block-dist centred {:center @center})
                                                                          :id :x :init [0.0])]
                                                          (swap! log conj x)
                                                          x)))
                                           (assoc F :method :nuts :iterations 600 :chains 2 :burn 300 :seed 7)))
         xs (map measure/get-value (measure/get-contexts posterior))]
     (is (< (Math/abs (- 3.0 (/ (reduce + xs) (count xs)))) 0.2))
     (is (= [] @log) "nothing reaches the caller"))))

(defn- lnorm [x mu s] (dist/logpdf (dist/normal mu s) x))

(def ^:private pt-block
  ;; θ = (a, b) ~ N(center, 0.3²) each; the reference N(0, 3²)
  (block/block {:block/id :pt :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               (let [lp (fn [[a b] {:keys [center]}] (+ (lnorm a center 0.3) (lnorm b center 0.3)))]
                 {:log-density (fn [th inputs] (lp (vec th) inputs))
                  :value+grad (fn [th inputs]
                                (let [th (vec th) f (lp th inputs) h 1e-6]
                                  [f (double-array (map (fn [i] (/ (- (lp (update th i + h) inputs) f) h)) (range 2)))]))
                  :sample (fn [_] [(* 3.0 (dist/draw (dist/normal 0.0 1.0))) (* 3.0 (dist/draw (dist/normal 0.0 1.0)))])
                  :sample-log-density (fn [th _] (let [[a b] (vec th)] (+ (lnorm a 0.0 3.0) (lnorm b 0.0 3.0))))})))

(deftest parallel-tempering-reads-the-caller-s-state
  (in-root
   (let [center (v2/atom 1.5)
         log (v2/atom [])
         posterior (infer! #(inference/parallel-tempering-infer
                           (fn [] (task (let [[a b] (sample (block/block-dist pt-block {:center @center})
                                                            :id :theta :init [0.0 0.0])]
                                          (swap! log conj :mine)
                                          {:a a :b b :log @log})))
                           6 (assoc F :rounds 8 :seed 3)))
         values (map measure/get-value (measure/get-contexts posterior))
         mean-a (/ (reduce + (map :a values)) (count values))]
     (is (< (Math/abs (- 1.5 mean-a)) 0.1) (str mean-a))
     (is (every? #(= [:mine] (:log %)) values) "every written-back state holds its own write")
     (is (= [] @log) "nothing reaches the caller"))))

;; -----------------------------------------------------------------------------
;; Refusals
;; -----------------------------------------------------------------------------

(def ^:private normal-mean
  (a/model [ys]
           (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
             (loop [i 0]
               (when (< i (count ys))
                 (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
                 (recur (inc i))))
             mu)))

(deftest compiled-rmh-refuses-world-options
  (in-root
   (doseq [opts [F {:world-opts {}}]]
     (let [error (try (inference/infer (fn [] (normal-mean [0.5 1.0]))
                                       (merge {:method :rmh :compiled? true :iterations 10 :chains 1} opts))
                      nil
                      (catch clojure.lang.ExceptionInfo e e))]
       (is (= ::inference/invalid-world-policy (:type (ex-data error))) (pr-str opts))))
   (testing ":world-policy :fresh is what compiled code does anyway"
     (is (some? (inference/infer (fn [] (normal-mean [0.5 1.0]))
                                 {:method :rmh :compiled? true :iterations 10 :chains 1
                                  :world-policy :fresh}))))))

(deftest smc2-refuses-fork
  (in-root
   (is (= ::smc2/unsupported-world-policy
          (:type (ex-data (try ((smc2/smc2 (fn [] (task nil)) {:params #{:theta} :world-policy :fork})
                                identity identity)
                               nil
                               (catch clojure.lang.ExceptionInfo e e))))))))

(deftest resource-authority-is-refused-on-engine-v2
  (in-root
   (doseq [[method run] {:smc #(inference/smc-infer (fn [] (writer (v2/atom []))) 2 %)
                         :pimh #(inference/pimh-infer (fn [] (writer (v2/atom []))) 2 2 %)}]
     (let [error (try (infer! #(run (assoc F :authority :ledger :grant {:tokens 1}))) nil
                      (catch Exception e e))]
       (is (some #(#{::canonical/unsupported-authority :org.replikativ.foerster.core/unsupported-authority}
                   (:type (ex-data %)))
                 (take-while some? (iterate ex-cause error)))
           (str method))))))
