(ns org.replikativ.foerster.world-particles-test
  "Canonical Yggdrasil worlds for effectful inference particles."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.proposal :as proposal]
            [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.runtime.async :refer [await-op]]
            [org.replikativ.foerster.runtime.execution :as executor]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.core :as inference]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.yggdrasil :as ygg]
            [yggdrasil.convergent.gset :as g]))

(defn- mem-gset [id]
  (g/gset id {:store-config {:backend :memory :id (random-uuid)}}
          {:sync? true}))

(defn- observed-model []
  (task
   (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
   :done))

(defn- await-cps [operation]
  (let [result (promise)]
    (operation #(deliver result [:ok %])
               #(deliver result [:error %]))
    (deref result 5000 ::timed-out)))

(deftest world-policy-is-explicit
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [result
              @(task
                (try
                  (await-op (inference/smc-infer
                          (fn [] (task :done)) 1 {:world-policy :unknown}))
                  :unexpected-success
                  (catch Throwable error error)))]
          (is (instance? Throwable result))
          (is (= ::inference/invalid-world-policy
                 (:type (ex-data result))))))
      (finally
        (v2/release-world! root)))))

(deftest pure-inference-has-no-world-descriptors
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [posterior @(task (await-op (inference/smc-infer (fn [] (task 7)) 2 {:world-policy :fresh})))]
          (is (= [7 7] (mapv measure/get-value (measure/get-contexts posterior))))
          (is (= [] (measure/world-descriptors posterior)))))
      (finally
        (v2/release-world! root)))))

;; -----------------------------------------------------------------------------
;; Canonical worlds on savepoint SMC
;; -----------------------------------------------------------------------------

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

(defn- slot-name []
  (keyword (str "p" (v2/get-in-world v2/*world* [:inference :slot]))))

(deftest smc-resamples-independent-worlds-and-discards-the-tree
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [knowledge (ygg/register! (-> (mem-gset "particle-kb")
                                           (g/conj :root {:sync? true})))
              model (task
                     (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                     (reset! (ygg/system-signal "particle-kb")
                             (g/conj @knowledge (slot-name) {:sync? true}))
                     (g/elements @knowledge {:sync? true}))]
          (capturing-scopes [scopes]
                            (let [posterior @(task (await-op (inference/smc-infer
                                                           model 4
                                                           {:world-policy :fork
                                            ;; resample even with equal weights
                                                            :resample-threshold 2.0})))
                                  results (mapv measure/get-value (measure/get-contexts posterior))
                                  descriptors (measure/world-descriptors posterior)]
                              (testing "each particle sees the caller's state and its own write"
                                (is (= #{#{:root :p0} #{:root :p1} #{:root :p2} #{:root :p3}} (set results))))
                              (testing "particle writes never reach the caller's world"
                                (is (= #{:root} (g/elements @knowledge {:sync? true}))))
                              (testing "the posterior names its worlds, settled"
                                (is (= 4 (count descriptors)))
                                (is (every? #(and (= :particle (:fork/purpose %))
                                                  (= :discarded (:fork/status %)))
                                            descriptors))
                                (is (every? #(instance? org.replikativ.foerster.measure.Sample %)
                                            (measure/get-contexts posterior))
                                    "no execution context is retained"))
                              (testing "every world is discarded before the posterior is delivered"
                                (is (= 1 (count @scopes)) "the root and the session's worlds share one")
                                (is (settled? @scopes)))))))
      (finally
        (v2/release-world! root)))))

(deftest every-lineage-ends-once
  (let [root (context/create-execution-context
              {:executor (executor/thread-pool-executor 4)})
        finalized (atom [])
        n 3]
    (try
      (binding [v2/*world* root]
        (let [model (task
                     (try
                       (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                       :done
                       (finally
                         (swap! finalized conj (:fork-id v2/*world*)))))
              result (deref (task (await-op (inference/smc-infer
                                          model n {:world-policy :fork
                                                   :executor (:executor root)
                                                   :resample-threshold 2.0})))
                            5000 ::timed-out)]
          (is (= (repeat n :done) (mapv measure/get-value (measure/get-contexts result))))
          ;; a resampled world continues in its copies instead of unwinding:
          ;; the model ends once per particle, in its final world
          (is (= n (count @finalized)))
          (is (apply distinct? @finalized) "each world reaches one terminal path")))
      (finally
        (v2/release-world! root)))))

(deftest every-particle-runs-the-whole-model
  ;; A canonical model's effects may be random without a sample site (a
  ;; model call): each particle makes its own, none shares another's prefix.
  (let [root (v2/world)
        calls (atom 0)]
    (try
      (binding [v2/*world* root]
        (let [posterior @(task (await-op (inference/smc-infer
                                       (fn [] (task
                                        (let [answer (swap! calls inc)]
                                          (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                          answer)))
                                       4 {:world-policy :fork})))]
          (is (= 4 @calls))
          (is (= #{1 2 3 4} (set (map measure/get-value (measure/get-contexts posterior)))))))
      (finally
        (v2/release-world! root)))))

(deftest cancelling-public-inference-spin-cancels-and-joins-worlds
  (let [root (context/create-execution-context
              {:executor (executor/thread-pool-executor 4)})
        task* (atom nil)
        model-entered (promise)
        model-cleaned? (atom false)
        discard-entered (promise)
        release-discard (promise)
        discard-world ygg/discard-fork!]
    (try
      (capturing-scopes [scopes]
                        (let [outcome
                              (future
                                (with-redefs
                                 [ygg/discard-fork!
                                  (fn [handle opts]
                                    (let [operation (discard-world handle opts)]
                                      (fn [resolve reject]
                                        (deliver discard-entered true)
                                        (future
                                          @release-discard
                                          (if (fn? operation) (operation resolve reject) (resolve operation))))))]
                                  (binding [v2/*world* root]
                                    (let [task (inference/smc-infer
                                                (fn [] (task
                                                 (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                                 (try
                                                   (deliver model-entered true)
                                                   (await-op (fn [_resolve _reject] nil))
                                                   (finally (reset! model-cleaned? true)))))
                                                2
                                                {:world-policy :fork
                                                 :executor (:executor root)})]
                                      (reset! task* task)
                                      (try @task (catch Throwable error error))))))]
                          (is (= true (deref model-entered 5000 ::timed-out)))
                          (binding [v2/*world* root]
                            (spin-core/cancel-spin! @task*))
                          (is (= true (deref discard-entered 5000 ::timed-out)))
                          (is (= ::still-waiting (deref outcome 100 ::still-waiting))
                              "public cancellation stays pending through asynchronous discard")
                          (deliver release-discard true)
                          (let [result (deref outcome 5000 ::timed-out)]
                            (is (instance? Throwable result))
                            (is (= spin-core/spin-cancelled (:type (ex-data result))))
                            (is @model-cleaned?)
                            (is (settled? @scopes)
                                "the public Spin rejects only after every world is discarded"))))
      (finally
        (deliver release-discard true)
        (v2/release-world! root)))))

(deftest cancellation-during-the-root-fork-joins-the-late-world
  (let [root (context/create-execution-context
              {:executor (executor/thread-pool-executor 2)})
        task* (atom nil)
        fork-entered (promise)
        release-fork (promise)
        model-ran? (atom false)
        fork-world ygg/fork!]
    (try
      (capturing-scopes [scopes]
                        (let [outcome
                              (future
                                (with-redefs
                                 [ygg/fork!
                                  (fn [opts]
                                    (let [operation (fork-world opts)]
                                      (fn [resolve reject]
                                        (operation
                                         (fn [handle]
                                           (deliver fork-entered true)
                                           (future @release-fork (resolve handle)))
                                         reject))))]
                                  (binding [v2/*world* root]
                                    (let [task (inference/smc-infer
                                                (fn [] (task (reset! model-ran? true) :done)) 1
                                                {:world-policy :fork
                                                 :executor (:executor root)})]
                                      (reset! task* task)
                                      (try @task (catch Throwable error error))))))]
                          (is (= true (deref fork-entered 5000 ::timed-out)))
                          (binding [v2/*world* root]
                            (spin-core/cancel-spin! @task*))
                          (is (= ::still-waiting (deref outcome 100 ::still-waiting))
                              "cancellation waits for the fork in flight")
                          (deliver release-fork true)
                          (let [result (deref outcome 5000 ::timed-out)]
                            (is (= spin-core/spin-cancelled (:type (ex-data result))))
                            (is (not @model-ran?))
                            (is (= 1 (count @scopes)))
                            (is (settled? @scopes) "the late root is discarded")
                            (is (= 1 (count (world-scope/descriptors (first @scopes))))))))
      (finally
        (deliver release-fork true)
        (v2/release-world! root)))))

(deftest a-copy-failure-fails-inference-and-leaves-no-world
  (doseq [failing [1 2]] ; the particles, a resampling
    (let [root (v2/world)
          copies (atom 0)
          copy-world ygg/copy-fork!]
      (try
        (binding [v2/*world* root]
          (capturing-scopes [scopes]
                            (let [result
                                  (with-redefs
                                   [ygg/copy-fork!
                                    (fn [handle k opts]
                                      (if (= failing (swap! copies inc))
                                        (throw (ex-info "synthetic copy failure" {:copy failing}))
                                        (copy-world handle k opts)))]
                                    (deref (task (try
                                                   (await-op (inference/smc-infer
                                                           (fn [] (observed-model)) 4
                                                           {:world-policy :fork :resample-threshold 2.0}))
                                                   (catch Throwable error error)))
                                           5000 ::timed-out))
                                  recovery (:world/recovery (ex-data result))]
                              (is (= ::inference/inference-failed (:type (ex-data result))) (str "copy " failing))
                              (is (= {:copy failing} (ex-data (ex-cause result))))
                              (is (map? recovery))
                              (is (= [:ok nil] (await-cps ((:discard! recovery)))) "cleanup is idempotent")
                              (is (settled? @scopes)))))
        (finally
          (v2/release-world! root))))))

(deftest model-failure-exposes-portable-world-recovery
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [result (deref (task
                             (try
                               (await
                                (inference/smc-infer
                                 (fn [] (task
                                  (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                  (throw (ex-info "model failed" {:stage :model}))))
                                 3
                                 {:world-policy :fork}))
                               :unexpected-success
                               (catch Throwable error error)))
                            5000 ::timed-out)
              recovery (:world/recovery (ex-data result))]
          (is (= ::inference/inference-failed (:type (ex-data result))))
          (is (= {:stage :model} (ex-data (ex-cause result))))
          (is (= 4 (count (:descriptors recovery))) "the root, then the particles")
          (is (every? #(= :particle (:fork/purpose %)) (:descriptors recovery)))
          (is (every? #(keyword? (:fork/id %)) (:descriptors recovery)))
          (is (instance? clojure.lang.IAtom (:manager recovery))
              "the live recovery capability stays process-local")
          (is (= [:ok nil] (await-cps (:await-quiescent recovery))))
          (is (= [:ok nil] (await-cps ((:discard! recovery)))))
          (is (= :discarded (:status @(:manager recovery))))))
      (finally
        (v2/release-world! root)))))

(defn- latent-model []
  (task
   (let [x (sample (dist/normal 0.0 1.0) :id :x)]
     (observe (dist/normal x 1.0) 0.5 :id :y)
     x)))

(defn- sweeps-of [posterior n]
  (partition n (mapv measure/get-value (measure/get-contexts posterior))))

(deftest conditional-sweeps-keep-their-trajectory-in-canonical-worlds
  ;; slot 0 of every particle Gibbs sweep follows the trajectory drawn from
  ;; the sweep before, so each sweep holds a value of the previous one (PGAS
  ;; redraws the retained particle's past, which here is all of it)
  (let [root (v2/world)
        n 4]
    (try
      (binding [v2/*world* root]
        (doseq [[method run] [[:pgibbs #(inference/pgibbs-infer (fn [] (latent-model)) n 3 {:world-policy :fork})]
                              [:pgas #(inference/pgas-infer (fn [] (latent-model)) n 3 {:world-policy :fork})]]]
          (let [posterior @(task (await-op (run)))
                sweeps (sweeps-of posterior n)]
            (is (= 3 (count sweeps)) (str method))
            (when (= :pgibbs method)
              (doseq [[before after] (partition 2 1 sweeps)]
                (is (seq (filter (set before) after)) "keeps the retained value")))
            (is (= (* 3 n) (count (measure/world-descriptors posterior))))
            (is (every? #(= :discarded (:fork/status %)) (measure/world-descriptors posterior))))))
      (finally
        (v2/release-world! root)))))

(deftest interacting-particle-mcmc-runs-in-canonical-worlds
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [posterior @(task (await-op (inference/ipmcmc-infer
                                       (fn [] (latent-model)) 3 2
                                       {:world-policy :fork :num-nodes 2 :num-csmc-nodes 1})))]
          (is (pos? (count (measure/get-contexts posterior))))
          (is (every? #(= :particle (:fork/purpose %)) (measure/world-descriptors posterior)))))
      (finally
        (v2/release-world! root)))))

(deftest a-proposal-decides-the-latent-sites
  (let [root (v2/world)
        ;; the value 42 with a density one nat above the site's prior: the
        ;; weight is log p(42) − log q(42) = −1, plus the observation
        fixed (proposal/from-fn (fn [{d :dist}] {:value 42.0 :log-q (+ 1.0 (dist/logpdf d 42.0))}))]
    (try
      (binding [v2/*world* root]
        (doseq [policy [:fresh :fork]]
          (let [posterior @(task (await-op (inference/importance-sampling (fn [] (latent-model)) 3
                                                                       {:world-policy policy
                                                                        :proposal fixed})))]
            (is (= [42.0 42.0 42.0] (mapv measure/get-value (measure/get-contexts posterior))))
            (is (every? #(< (Math/abs (- % (+ -1.0 (dist/logpdf (dist/normal 42.0 1.0) 0.5)))) 1e-9)
                        (mapv second (measure/get-particles posterior)))
                "the proposal's correction and the observation make the weight"))))
      (finally
        (v2/release-world! root)))))

(defn- wallet-authority
  "A ledger outside every world: {world-id {:wallet {resource n} :from id}}."
  [ledger]
  (reify world-scope/PResourceAuthority
    (grant! [_ source child grant]
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

(deftest particles-split-the-inference-s-budget
  (let [root (v2/world)
        ledger (atom {(:fork-id root) {:wallet {:tokens 100}}})
        tokens #(get-in @ledger [(:fork-id v2/*world*) :wallet :tokens])]
    (try
      (binding [v2/*world* root]
        (let [posterior @(task (await-op (inference/smc-infer
                                       (fn [] (task
                                        (let [at-start (tokens)]
                                          (observe (dist/normal 0.0 1.0) 0.0 :id :evidence)
                                          [at-start (tokens)])))
                                       4 {:world-policy :fork
                                          :authority (wallet-authority ledger)
                                          :grant {:tokens 12}
                                          :resample-threshold 2.0})))
              budgets (mapv measure/get-value (measure/get-contexts posterior))]
          (is (every? #(= 3 (first %)) budgets) "each particle starts with an even share")
          (is (every? #(<= (second %) 3) budgets) "a resampled world's share is split among its copies")
          (is (= {(:fork-id root) {:wallet {:tokens 100}}} @ledger)
              "every world gives back what it has left; nothing is multiplied")))
      (finally
        (v2/release-world! root)))))

(deftest a-world-that-may-not-be-copied-is-not-made-into-particles
  (let [root (v2/world)
        ran? (atom false)]
    (try
      (binding [v2/*world* root]
        (ygg/register! (mem-gset "live-handle") {:grade :affine})
        (capturing-scopes [scopes]
                          (let [result (deref (task (try
                                                      (await-op (inference/smc-infer
                                                              (fn [] (task (reset! ran? true) :done)) 3
                                                              {:world-policy :fork}))
                                                      (catch Throwable error error)))
                                              5000 ::timed-out)]
                            (is (= ::inference/inference-failed (:type (ex-data result))))
                            (is (= ::ygg/copy-forbidden (:type (ex-data (ex-cause result)))))
                            (is (not @ran?) "refused before the model runs")
                            (is (settled? @scopes)))))
      (finally
        (v2/release-world! root)))))

;; -----------------------------------------------------------------------------
;; Importance sampling, PIMH and BBVI in canonical worlds
;; -----------------------------------------------------------------------------

(deftest particle-methods-see-the-caller-s-state-in-isolated-worlds
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [log (sa/atom [:root])
              model (fn []
                      (task
                       (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                         (observe (dist/normal x 1.0) 0.5 :id :y)
                         (swap! log conj :mine)
                         @log)))]
          (doseq [[method run] [[:importance #(inference/importance-sampling (fn [] (model)) 4 {:world-policy :fork})]
                                [:pimh #(inference/pimh-infer (fn [] (model)) 3 2 {:world-policy :fork})]
                                [:bbvi #(inference/bbvi-infer (fn [] (model)) 4 2 {:world-policy :fork})]]]
            (capturing-scopes [scopes]
                              (let [posterior @(task (await-op (run)))
                                    descriptors (measure/world-descriptors posterior)]
                                (is (every? #{[:root :mine]} (mapv measure/get-value (measure/get-contexts posterior)))
                                    (str method ": each particle sees the caller's state and its own write only"))
                                (is (= [:root] @log) (str method ": nothing reaches the caller's world"))
                                (is (= (count (measure/get-particles posterior)) (count descriptors))
                                    (str method ": every particle names its world"))
                                (is (every? #(and (= :particle (:fork/purpose %)) (= :discarded (:fork/status %)))
                                            descriptors)
                                    (str method))
                                (is (seq @scopes) (str method))
                                (is (settled? @scopes) (str method ": every world is discarded")))))))
      (finally
        (v2/release-world! root)))))

;; -----------------------------------------------------------------------------
;; Exact enumeration in canonical worlds
;; -----------------------------------------------------------------------------

(def ^:private backlog {:free-at 1.0 :jobs [{:id :in-flight :finish 1.0}]})

(def ^:private readings [{:id :a :arrival 0.5 :work 1.0 :reported 1.9}
                         {:id :b :arrival 1.2 :work 0.7 :reported 2.7}
                         {:id :c :arrival 3.0 :work 1.2 :reported 4.1}])

(defn- serve!
  "Serve `job` on the queue held by the runtime atom `queue`; its finishing time."
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

(deftest enumeration-runs-on-the-caller-s-state
  (let [root (v2/world)
        [p-fast log-z] (queue-reference)]
    (try
      (binding [v2/*world* root]
        (let [queue (sa/atom backlog)]
          (capturing-scopes [scopes]
                            (let [posterior @(task (await-op (inference/infer (fn [] (queue-model queue))
                                                                           {:method :enumerate :world-policy :fork})))
                                  values (mapv measure/get-value (measure/get-contexts posterior))
                                  descriptors (measure/world-descriptors posterior)]
                              (is (< (Math/abs (- p-fast 0.5659282411891152)) 1e-12) "the reference")
                              (is (< (Math/abs (- p-fast (:mean (inference/query posterior #(if (= 1.4 (:speed %)) 1.0 0.0)))))
                                     1e-12))
                              (is (< (Math/abs (- log-z (measure/log-marginal posterior))) 1e-12))
                              (testing "every branch served on the caller's backlog, in a queue of its own"
                                (is (= 16 (count values)))
                                (is (every? #(= (:jobs backlog) (subvec (:jobs (:queue %)) 0 1)) values))
                                (is (every? #(= [:in-flight :a :b :c] (mapv :id (:jobs (:queue %)))) values))
                                (is (= 16 (count (set (map :queue values))))))
                              (is (= backlog @queue) "the caller's queue is unchanged")
                              (testing "the outcomes name their worlds, all discarded before delivery"
                                (is (= 16 (count descriptors)))
                                (is (every? #(and (= :particle (:fork/purpose %)) (= :discarded (:fork/status %)))
                                            descriptors))
                                (is (= 1 (count @scopes)))
                                (is (settled? @scopes)))))))
      (finally
        (v2/release-world! root)))))

(deftest enumeration-branches-write-in-isolation
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [log (sa/atom [:root])
              posterior @(task (await-op (inference/infer
                                       (fn [] (task
                                        (let [x (sample (dist/flip 0.5) :id :x)]
                                          (swap! log conj x)
                                          (let [y (sample (dist/categorical {:a 0.2 :b 0.8}) :id :y)]
                                            (swap! log conj y)
                                            @log))))
                                       {:method :enumerate :world-policy :fork})))
              ps (measure/get-particles posterior)]
          (is (= #{[:root true :a] [:root true :b] [:root false :a] [:root false :b]}
                 (set (map (comp measure/get-value first) ps)))
              "each branch sees its own writes and no other branch's")
          (is (= [:root] @log))
          (is (< (Math/abs (measure/log-marginal posterior)) 1e-12))))
      (finally
        (v2/release-world! root)))))

(deftest enumeration-branches-split-the-inference-s-budget
  (let [root (v2/world)
        ledger (atom {(:fork-id root) {:wallet {:tokens 100}}})
        tokens #(get-in @ledger [(:fork-id v2/*world*) :wallet :tokens])]
    (try
      (binding [v2/*world* root]
        (let [posterior @(task (await-op (inference/infer
                                       (fn [] (task
                                        (let [at-start (tokens)]
                                          (sample (dist/categorical {:a 1.0 :b 1.0 :c 1.0}) :id :x)
                                          [at-start (tokens)])))
                                       {:method :enumerate :world-policy :fork
                                        :authority (wallet-authority ledger) :grant {:tokens 12}})))]
          (is (= (repeat 3 [12 4]) (map (comp measure/get-value first) (measure/get-particles posterior)))
              "the model starts with the grant; three branches share it evenly")
          (is (= {(:fork-id root) {:wallet {:tokens 100}}} @ledger) "every branch gives back what it has left")))
      (finally
        (v2/release-world! root)))))

(deftest enumeration-refuses-a-world-that-may-not-be-copied
  (let [root (v2/world)
        ran? (atom false)]
    (try
      (binding [v2/*world* root]
        (ygg/register! (mem-gset "live-handle-enumeration") {:grade :affine})
        (capturing-scopes [scopes]
                          (let [result (deref (task (try
                                                      (await-op (inference/infer
                                                              (fn [] (task (reset! ran? true)
                                                                    (sample (dist/flip 0.5) :id :x)))
                                                              {:method :enumerate :world-policy :fork}))
                                                      (catch Throwable error error)))
                                              5000 ::timed-out)]
                            (is (= ::inference/inference-failed (:type (ex-data result))))
                            (is (= ::ygg/copy-forbidden (:type (ex-data (ex-cause result)))))
                            (is (not @ran?) "refused before the model runs")
                            (is (settled? @scopes)))))
      (finally
        (v2/release-world! root)))))

(deftest enumeration-options-are-checked
  (let [root (v2/world)]
    (try
      (binding [v2/*world* root]
        (let [model (task (sample (dist/flip 0.3) :id :x))
              fresh @(task (await-op (inference/infer model {:method :enumerate :world-policy :fresh})))]
          (is (= [] (measure/world-descriptors fresh)) "fresh worlds name no worlds")
          (is (< (Math/abs (measure/log-marginal fresh)) 1e-12))
          (doseq [opts [{:resample-threshold 0.5} {:grant {:tokens 1}} {:world-policy :fresh :world-opts {}}]]
            (is (thrown? clojure.lang.ExceptionInfo
                         (inference/infer model (assoc opts :method :enumerate)))
                (str opts)))))
      (finally
        (v2/release-world! root)))))
