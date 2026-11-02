(ns org.replikativ.foerster.fork-policy-test
  "`:world-policy :fork` beyond savepoint SMC: one frozen fork of the
  caller's world per inference, a world and scope of its own for every run
  (sweep, chain, particle, iteration), no incremental reuse under :fork, and
  refusals where a method has no worlds."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.world.scope :as world-scope]))

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
           smc (var-get #'org.replikativ.foerster.smc/smc)]
       (capturing-scopes [scopes]
                         (with-redefs [org.replikativ.foerster.smc/smc
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
