(ns org.replikativ.foerster.fork-policy-test
  "`:world-policy :fork` beyond savepoint SMC: one frozen fork of the
  caller's world per inference, a world and scope of its own for every run
  (sweep, chain, particle, iteration), no incremental reuse under :fork, and
  refusals where a method has no worlds."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.kernel :as kernel]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.smc2 :as smc2]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.keyed :refer [keyed]]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.yggdrasil :as ygg]
            [yggdrasil.convergent.gset :as g]))

(defmacro ^:private capturing-scopes
  "Run `body` with every world scope created meanwhile collected in `scopes`."
  [[scopes] & body]
  `(let [~scopes (atom [])
         create# world-scope/create]
     (with-redefs [world-scope/create (fn [opts#]
                                        (let [s# (create# opts#)]
                                          (swap! ~scopes conj s#)
                                          s#))]
       ~@body)))

(defn- settled? [scopes]
  (every? #(and (= :discarded (:status @%)) (empty? (:handles @%))) scopes))

(defmacro ^:private in-root
  "Run `body` with a fresh caller world bound, stopped afterwards."
  [& body]
  `(let [root# (context/create-execution-context)]
     (try
       (binding [ec/*execution-context* root#] ~@body)
       (finally (context/stop-context! root#)))))

(defn- failure [run]
  (deref (spin (try (await (run)) :unexpected-success
                    (catch Throwable error error)))
         60000 ::timed-out))

;; -----------------------------------------------------------------------------
;; Compiled paths have no worlds
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
   (doseq [opts [{:world-policy :fork} {:world-opts {}} {:authority :ledger :grant {:tokens 1}}]]
     (let [error (try (inference/infer (normal-mean [0.5 1.0])
                                       (merge {:method :rmh :compiled? true :iterations 10 :chains 1} opts))
                      nil
                      (catch clojure.lang.ExceptionInfo e e))]
       (is (= ::inference/invalid-world-policy (:type (ex-data error))) (pr-str opts))))
   (testing ":world-policy :fresh is what compiled code does anyway"
     (is (some? (inference/infer (normal-mean [0.5 1.0])
                                 {:method :rmh :compiled? true :iterations 10 :chains 1
                                  :world-policy :fresh}))))))

;; -----------------------------------------------------------------------------
;; Batched SMC in canonical worlds
;; -----------------------------------------------------------------------------

(deftest batched-smc-runs-in-canonical-worlds
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (in-root
       (let [log (sa/atom [:root])
             model (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                           (observe (dist/normal x 1.0) 0.5 :id :y)
                           (observe (dist/normal x 1.0) 0.7 :id :z)
                           (swap! log conj :mine)
                           [x @log]))]
         (capturing-scopes [scopes]
                           (let [posterior @(spin (await (inference/smc-infer
                                                          model 400 {:world-policy :fork :batch 16 :executor exec
                                                                     :resample-threshold 1.0 :seed 3})))
                                 values (mapv measure/get-value (measure/get-contexts posterior))
                                 ws (mapv #(Math/exp (second %)) (measure/get-particles posterior))
                                 mean (/ (reduce + (map * (map first values) ws)) (reduce + ws))]
                             (is (= 400 (count values)))
                             (is (every? #{[:root :mine]} (map second values))
                                 "every particle sees the caller's state and its own write only")
                             (is (= [:root] @log) "nothing reaches the caller")
                             (is (< (Math/abs (- mean 0.4)) 0.15) (str mean " vs 0.4"))
                             (is (some #(<= (:size %) 16) (:history posterior)) "resampled in batches")
                             (is (every? #(= :discarded (:fork/status %)) (measure/world-descriptors posterior)))
                             (is (settled? @scopes))))))
      (finally (.close ^java.lang.AutoCloseable exec)))))

;; -----------------------------------------------------------------------------
;; Methods of several runs fork the caller once
;; -----------------------------------------------------------------------------

(defn- latent-reader
  "A model that reads the caller's `state` and writes its own `log`."
  [state log]
  (spin
   (let [x (sample (dist/normal 0.0 1.0) :id :x)]
     (observe (dist/normal x 1.0) 0.5 :id :y)
     (swap! log conj :mine)
     [@state @log])))

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
     (let [caller ec/*execution-context*
           state (sa/atom :before)
           log (sa/atom [:root])
           runs (atom 0)
           smc (var-get #'smc/smc)]
       (capturing-scopes [scopes]
                         (with-redefs [smc/smc
                                       (fn [& args]
                                         (when (= 2 (swap! runs inc))
                                           (binding [ec/*execution-context* caller]
                                             (reset! state :changed)))
                                         (apply smc args))]
                           (let [posterior @(spin (await (run (latent-reader state log) {:world-policy :fork})))
                                 values (mapv measure/get-value (measure/get-contexts posterior))]
                             (is (> @runs 1) (str method ": several runs"))
                             (is (= :changed @state) (str method ": the caller changed meanwhile"))
                             (is (every? #{[:before [:root :mine]]} values)
                                 (str method ": every run sees the state as inference started, and its own write"))
                             (is (= [:root] @log) (str method ": nothing reaches the caller"))
                             (is (every? #(= :discarded (:fork/status %)) (measure/world-descriptors posterior))
                                 (str method))
                             (is (settled? @scopes) (str method ": every world, the frozen fork included, is discarded")))))))))

(defn- wallet-authority
  "A ledger outside every world: {world-id {:wallet {resource n} :from id}};
  `grants` counts the grants each source world made."
  [ledger grants]
  (reify world-scope/PResourceAuthority
    (grant! [_ source child grant]
      (swap! grants update (:fork-id source) (fnil inc 0))
      (swap! ledger
             (fn [book]
               (let [left (merge-with - (get-in book [(:fork-id source) :wallet]) grant)]
                 (when (some neg? (vals left))
                   (throw (ex-info "Insufficient funds" {})))
                 (-> book
                     (assoc-in [(:fork-id source) :wallet] left)
                     (assoc (:fork-id child) {:wallet grant :from (:fork-id source)})))))
      nil)
    (return! [_ context]
      (swap! ledger
             (fn [book]
               (if-let [{:keys [wallet from]} (get book (:fork-id context))]
                 (-> book
                     (update-in [from :wallet] #(merge-with + % wallet))
                     (dissoc (:fork-id context)))
                 book)))
      nil)
    (balance [_ context] (get-in @ledger [(:fork-id context) :wallet]))))

(deftest a-multi-run-inference-draws-its-grant-once
  (doseq [[method run share] [[:pimh #(inference/pimh-infer %1 4 3 %2) 3]
                              [:ipmcmc #(inference/ipmcmc-infer %1 2 2 (assoc %2 :num-nodes 2 :num-csmc-nodes 1)) 3]]]
    (in-root
     (let [root ec/*execution-context*
           ledger (atom {(:fork-id root) {:wallet {:tokens 100}}})
           grants (atom {})
           tokens #(get-in @ledger [(:fork-id ec/*execution-context*) :wallet :tokens])
           posterior @(spin (await (run (spin (let [at-start (tokens)]
                                                (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                                at-start))
                                        {:world-policy :fork :authority (wallet-authority ledger grants)
                                         :grant {:tokens 12}})))]
       (is (every? #(= share %) (map measure/get-value (measure/get-contexts posterior)))
           (str method ": every particle of every run starts with its share of the one grant"))
       (is (= 1 (get @grants (:fork-id root))) (str method ": the caller's wallet is drawn once"))
       (is (= {(:fork-id root) {:wallet {:tokens 100}}} @ledger)
           (str method ": every world gives back what it has left"))))))

(deftest a-failing-run-exposes-recovery-and-discards-every-world
  (in-root
   (capturing-scopes [scopes]
                     (let [runs (atom 0)
                           result (failure #(inference/pimh-infer
                                             (spin
                                              (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                              (when (= 4 (swap! runs inc))
                                                (throw (ex-info "model failed" {:stage :model})))
                                              :ok)
                                             3 3 {:world-policy :fork}))]
                       (is (= ::inference/inference-failed (:type (ex-data result))))
                       (is (some? (:world/recovery (ex-data result))))
                       (is (= {:stage :model} (ex-data (ex-cause result))))
                       (is (settled? @scopes))))))

;; -----------------------------------------------------------------------------
;; Markov chains in forks of the caller's world
;; -----------------------------------------------------------------------------

(def ^:private backlog {:free-at 1.0 :jobs [{:id :in-flight :finish 1.0}]})

(def ^:private readings [{:id :a :arrival 0.5 :work 1.0 :reported 1.9}
                         {:id :b :arrival 1.2 :work 0.7 :reported 2.7}
                         {:id :c :arrival 3.0 :work 1.2 :reported 4.1}])

(def ^:private p-fast
  "P(speed = 1.4 | y) on the caller's backlog: the exact enumeration of
  `world-particles-test` (and the paper's independent sum)."
  0.5659282411891152)

(defn- serve! [queue {:keys [id arrival work]} speed jitter]
  (let [start (max arrival (:free-at @queue))
        finish (+ start (/ (* work jitter) speed))]
    (swap! queue (fn [state]
                   (-> state
                       (assoc :free-at finish)
                       (update :jobs conj {:id id :start start :finish finish}))))
    finish))

(defn- queue-model
  "The paper's queue, served on the caller's `queue`: every job's
  completion depends on what the model wrote for the jobs before it."
  [queue]
  (spin
   (let [speed (sample (dist/categorical {0.8 0.5, 1.4 0.5}) :id :speed)]
     (doseq [{:keys [id reported] :as job} readings]
       (let [jitter (sample (dist/categorical {0.8 0.5, 1.2 0.5}) :id [:jitter id])
             finish (serve! queue job speed jitter)]
         (observe (dist/normal finish 0.35) reported :id [:completion id])))
     {:speed speed :queue @queue})))

(defn- fast-fraction [posterior]
  (let [values (map measure/get-value (measure/get-contexts posterior))]
    (/ (count (filter #(= 1.4 (:speed %)) values)) (double (count values)))))

(def ^:private gibbs-sweep
  (kernel/cycle 600 (mapv #(kernel/finite-gibbs-kernel 1 %) [:speed [:jitter :a] [:jitter :b] [:jitter :c]])
                {:samples :all :burn 20}))

(deftest markov-chains-run-on-the-caller-s-state
  (doseq [[method run] [[:mh #(inference/infer % {:method :mh :iterations 2500 :chains 4 :burn 100
                                                  :world-policy :fork :seed %2})]
                        [:gibbs #(inference/kernel-infer % gibbs-sweep 4 {:world-policy :fork :seed %2})]]
          seed [1 2]]
    (in-root
     (let [queue (sa/atom backlog)]
       (capturing-scopes [scopes]
                         (let [posterior @(spin (await (run (queue-model queue) seed)))
                               values (map measure/get-value (measure/get-contexts posterior))]
                           (is (< (Math/abs (- p-fast (fast-fraction posterior))) 0.06)
                               (str method " " seed ": " (fast-fraction posterior) " vs exact " p-fast))
                           (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values)
                               (str method ": every state served the caller's backlog, once"))
                           (is (= backlog @queue) (str method ": the caller's queue is unchanged"))
                           (is (settled? @scopes) (str method ": every chain's worlds are discarded"))))))))

(def ^:private centred
  (block/block {:block/id :centred :block/target :complete-conditional
                :block/latents [{:name :x :shape [] :support :real}]}
               {:log-density (fn [th {:keys [center]}] (let [d (- (aget ^doubles th 0) center)] (* -0.5 d d)))
                :value+grad (fn [th {:keys [center]}]
                              (let [d (- (aget ^doubles th 0) center)]
                                [(* -0.5 d d) (double-array [(- d)])]))}))

(deftest nuts-reads-the-caller-s-state-as-block-inputs
  (in-root
   (let [center (sa/atom 3.0)
         log (sa/atom [])
         posterior @(spin (await (inference/infer (spin (let [[x] (sample (block/block-dist centred {:center @center})
                                                                          :id :x :init [0.0])]
                                                          (swap! log conj x)
                                                          x))
                                                  {:method :nuts :iterations 600 :chains 2 :burn 300
                                                   :world-policy :fork :seed 7})))
         xs (map measure/get-value (measure/get-contexts posterior))]
     (is (< (Math/abs (- 3.0 (/ (reduce + xs) (count xs)))) 0.2))
     (is (= [] @log) "nothing reaches the caller"))))

(deftest chains-refuse-authority-for-now
  (in-root
   (is (= ::inference/invalid-world-policy
          (:type (ex-data (failure #(inference/infer (queue-model (sa/atom backlog))
                                                     {:method :mh :iterations 10 :world-policy :fork
                                                      :authority :ledger :grant {:tokens 1}}))))))))

;; A keyed model that writes state: under :fresh, an MH move on a keyed site
;; re-runs its key in the world the trace ENDED in (`spindel.trace/update`),
;; so the key's write is applied again on top of its earlier one, and the
;; parent's continuation reads the doubled state (observed: 3 to 23 writes
;; per state of a 3-key model). Under :fork reuse is off and every state
;; holds exactly its own three writes.

(def ^:private keyed-reports [0.9 0.6 1.1])

(defn- keyed-model []
  (spin
   (let [log (sa/atom [])
         speed (sample (dist/categorical {0.8 0.5, 1.4 0.5}) :id :speed)
         _ (await (keyed :jobs [0 1 2]
                         (fn [i]
                           (spin (let [jitter (sample (dist/categorical {0.8 0.5, 1.2 0.5}) :id [:jitter i])]
                                   (swap! log conj i)
                                   (observe (dist/normal (/ jitter speed) 0.35) (nth keyed-reports i) :id [:obs i])
                                   jitter)))))]
     {:speed speed :writes (count @log)})))

(def ^:private keyed-p-fast
  (let [lik (fn [speed] (reduce * (map (fn [y] (reduce + (map #(* 0.5 (Math/exp (dist/logpdf (dist/normal (/ % speed) 0.35) y)))
                                                                   [0.8 1.2])))
                                       keyed-reports)))]
    (/ (lik 1.4) (+ (lik 0.8) (lik 1.4)))))

(deftest keyed-models-that-write-state-stay-exact-under-fork
  (in-root
   (let [posterior @(spin (await (inference/infer (keyed-model)
                                                  {:method :mh :iterations 2500 :chains 4 :burn 100
                                                   :world-policy :fork :seed 11})))
         values (map measure/get-value (measure/get-contexts posterior))]
     (is (every? #(= 3 (:writes %)) values) "every state holds its own writes, once")
     (is (< (Math/abs (- keyed-p-fast (fast-fraction posterior))) 0.06)
         (str (fast-fraction posterior) " vs exact " keyed-p-fast)))))

(deftest cancelling-a-chain-joins-and-discards-its-worlds
  (let [root (context/create-execution-context {:executor (executor/thread-pool-executor 2)})
        task* (atom nil)
        runs (atom 0)
        entered (promise)
        cleaned? (atom false)]
    (try
      (capturing-scopes [scopes]
                        (let [outcome (future
                                        (binding [ec/*execution-context* root]
                                          (let [task (inference/infer
                                                      (spin
                                                       (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                                         (observe (dist/normal x 1.0) 0.5 :id :y)
                                                         (when (= 5 (swap! runs inc))
                                                           (try
                                                             (deliver entered true)
                                                             (await (fn [_resolve _reject] nil))
                                                             (finally (reset! cleaned? true))))
                                                         x))
                                                      {:method :mh :iterations 50 :chains 1
                                                       :world-policy :fork :executor (:executor root)})]
                                            (reset! task* task)
                                            (try @task (catch Throwable error error)))))]
                          (is (= true (deref entered 5000 ::timed-out)))
                          (binding [ec/*execution-context* root]
                            (spin-core/cancel-spin! @task*))
                          (let [result (deref outcome 10000 ::timed-out)]
                            (is (instance? Throwable result))
                            (is (= spin-core/spin-cancelled (:type (ex-data result))))
                            (is @cleaned? "the blocked move unwound in its own world")
                            (is (settled? @scopes) "the frozen fork and every chain world are discarded"))))
      (finally
        (context/close-context! root)))))

;; -----------------------------------------------------------------------------
;; Tempered SMC, parallel tempering and inference programs
;; -----------------------------------------------------------------------------

(defn- weighted-fast [posterior]
  (let [ps (measure/get-particles posterior)
        ws (mapv #(Math/exp (second %)) ps)]
    (/ (reduce + (map (fn [[s _] w] (if (= 1.4 (:speed (measure/get-value s))) w 0.0)) ps ws))
       (reduce + ws))))

(deftest tempered-smc-runs-on-the-caller-s-state
  (in-root
   (let [queue (sa/atom backlog)]
     (capturing-scopes [scopes]
                       (let [posterior @(spin (await (inference/tempered-infer (queue-model queue) 300
                                                                               {:world-policy :fork :seed 5})))
                             values (map measure/get-value (measure/get-contexts posterior))]
                         (is (< (Math/abs (- p-fast (weighted-fast posterior))) 0.08)
                             (str (weighted-fast posterior) " vs exact " p-fast))
                         (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values))
                         (is (= backlog @queue) "the caller's queue is unchanged")
                         (is (settled? @scopes) "every particle's world is discarded"))))))

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
   (let [center (sa/atom 1.5)
         log (sa/atom [])]
     (capturing-scopes [scopes]
                       (let [posterior @(spin (await (inference/parallel-tempering-infer
                                                      (spin (let [[a b] (sample (block/block-dist pt-block {:center @center})
                                                                                :id :theta :init [0.0 0.0])]
                                                              (swap! log conj :mine)
                                                              {:a a :b b :log @log}))
                                                      6 {:world-policy :fork :rounds 8 :seed 3})))
                             values (map measure/get-value (measure/get-contexts posterior))
                             mean-a (/ (reduce + (map :a values)) (count values))]
                         (is (< (Math/abs (- 1.5 mean-a)) 0.1) (str mean-a))
                         (is (every? #(= [:mine] (:log %)) values) "every written-back state holds its own write")
                         (is (= [] @log) "nothing reaches the caller")
                         (is (settled? @scopes)))))))

(deftest inference-programs-run-trace-targets-on-the-caller-s-state
  (in-root
   (let [queue (sa/atom backlog)]
     (capturing-scopes [scopes]
                       (let [posterior @(spin (await (ip/run {:seed 42 :world-policy :fork}
                                                             (fn [scope]
                                                               (spin
                                                                ;; importance weights, then moves on the
                                                                ;; weighted population; a separate small
                                                                ;; run below resamples (copies) too
                                                                (let [p (await (ip/initialize scope (ip/trace-target (queue-model queue)
                                                                                                                     {:world-policy :fork})
                                                                                              600))
                                                                      p (ip/advance scope p (ip/temperature (:target p) 1.0))
                                                                      p (await (ip/move scope p (kernel/single-site-mh-kernel 2)))]
                                                                  (ip/finish scope p)))))))
                             values (map measure/get-value (measure/get-contexts posterior))]
                         (is (< (Math/abs (- p-fast (weighted-fast posterior))) 0.08)
                             (str (weighted-fast posterior) " vs exact " p-fast))
                         (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values))
                         (is (= backlog @queue))
                         (is (settled? @scopes)))))
   (testing "resampling copies particles in their canonical worlds"
     (let [queue (sa/atom backlog)]
       (capturing-scopes [scopes]
                         (let [posterior @(spin (await (ip/run {:seed 7 :world-policy :fork}
                                                               (fn [scope]
                                                                 (spin
                                                                  (let [p (await (ip/initialize scope (ip/trace-target (queue-model queue)
                                                                                                                       {:world-policy :fork})
                                                                                                40))
                                                                        p (ip/advance scope p (ip/temperature (:target p) 1.0))
                                                                        p (await (ip/resample scope p))
                                                                        p (await (ip/move scope p (kernel/single-site-mh-kernel 2)))]
                                                                    (ip/finish scope p)))))))]
                           (is (= 40 (count (measure/get-particles posterior))))
                           (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue (measure/get-value %)))))
                                       (measure/get-contexts posterior)))
                           (is (= backlog @queue))
                           (is (settled? @scopes))))))
   (testing "a :fork target needs a :fork program"
     (is (thrown-with-msg? Exception #"world-policy :fork"
                           (deref (spin (await (ip/run {:seed 1}
                                                       (fn [scope]
                                                         (spin (ip/finish scope (await (ip/initialize scope (ip/trace-target (queue-model (sa/atom backlog))
                                                                                                                             {:world-policy :fork})
                                                                                                      4))))))))
                                  10000 ::timed-out))))))

;; -----------------------------------------------------------------------------
;; PMMH
;; -----------------------------------------------------------------------------

(deftest pmmh-runs-every-iteration-on-the-caller-s-state
  ;; θ = the server's speed, the jitters filtered by SMC: P(speed = 1.4 | y)
  (in-root
   (let [queue (sa/atom backlog)
         model (fn []
                 (spin
                  (let [speed (sample (dist/categorical {0.8 0.5, 1.4 0.5}) :id :speed)]
                    (doseq [{:keys [id reported] :as job} readings]
                      (let [jitter (sample (dist/categorical {0.8 0.5, 1.2 0.5}) :id [:jitter id])
                            finish (serve! queue job speed jitter)]
                        (observe (dist/normal finish 0.35) reported :id [:completion id])))
                    {:speed speed :queue @queue})))]
     (capturing-scopes [scopes]
                       (let [posterior @(spin (await (inference/infer (model)
                                                                      {:method :pmmh :particles 60 :iterations 600 :burn 50
                                                                       :params #{:speed}
                                                                       :propose (fn [{:keys [speed]}]
                                                                                  {:theta {:speed (if (= speed 0.8) 1.4 0.8)}
                                                                                   :log-ratio 0.0})
                                                                       :world-policy :fork :seed 21})))
                             values (map measure/get-value (measure/get-contexts posterior))]
                         (is (< (Math/abs (- p-fast (fast-fraction posterior))) 0.1)
                             (str (fast-fraction posterior) " vs exact " p-fast))
                         (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values))
                         (is (= backlog @queue) "the caller's queue is unchanged")
                         (is (settled? @scopes) "every iteration's worlds are discarded"))))))

(deftest smc2-still-refuses-fork
  (in-root
   (is (= ::smc2/unsupported-world-policy
          (:type (ex-data (try ((smc2/smc2 (spin nil) {:params #{:theta} :world-policy :fork})
                                identity identity)
                               nil
                               (catch clojure.lang.ExceptionInfo e e))))))))

(deftest a-world-that-may-not-be-copied-is-refused-by-every-method
  (doseq [[method run] {:mh #(inference/infer % {:method :mh :iterations 5 :chains 1 :world-policy :fork})
                        :pimh #(inference/pimh-infer % 2 2 {:world-policy :fork})
                        :tempered #(inference/tempered-infer % 4 {:world-policy :fork})}]
    (in-root
     (ygg/register!
      (g/gset (str "live-handle-" (name method))
                                      {:store-config {:backend :memory :id (random-uuid)}} {:sync? true})
      {:grade :affine})
     (let [ran? (atom false)]
       (capturing-scopes [scopes]
                         (let [result (failure #(run (spin (reset! ran? true)
                                                           (sample (dist/normal 0.0 1.0) :id :x))))]
                           (is (= ::inference/inference-failed (:type (ex-data result))) (str method))
                           (is (some #(= ::ygg/copy-forbidden (:type (ex-data %)))
                                     (take-while some? (iterate ex-cause result)))
                               (str method))
                           (is (not @ran?) (str method ": refused before the model runs"))
                           (is (settled? @scopes) (str method))))))))
