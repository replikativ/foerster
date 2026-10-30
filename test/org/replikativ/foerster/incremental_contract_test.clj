(ns org.replikativ.foerster.incremental-contract-test
  (:require [clojure.test :refer [deftest is]]
            [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as it]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.keyed :refer [keyed]]
            [org.replikativ.spindel.trace :as trace]))

(defn- execute [make body]
  (b/run-infer 509
               #(pcps/async
                 (let [t (pcps/await (gfi/run-policy (make) (it/policy {:init? true}) {:seed 509}))]
                   (try (pcps/await (body t))
                        (finally (pcps/await (gfi/close! t))))))))

(defn- branch-child []
  (spin
   (let [x (sample (d/flip 0.5) :id :branch :init true)]
     (when x
       (aw/await (spin (let [z (sample (d/flip 0.7) :id :nested-z :init true)]
                         (observe (d/flip (if z 0.9 0.2)) true :id :nested-y)))))
     x)))

(deftest removal-accounting-includes-nested-spin-deaths
  (let [{:keys [delta absent? full-ratio changed-ratio result source]}
        (execute #(spin (aw/await (keyed :items [0] (fn [_] (branch-child)))))
                 (fn [old]
                   (pcps/async
                    (let [next (pcps/await (it/reexecute old #{:branch}
                                                         (it/policy {:keep? true :draw (fn [s _]
                                                                                         (when (= :branch (:savepoint/address s))
                                                                                           {:value false :symmetric? true}))})
                                                         {:seed 511}))
                          delta (::it/changed next)]
                      {:delta delta :absent? (not (contains? (:trace/entries next) :nested-y))
                       :full-ratio (it/mh-log-ratio old next (constantly 0.0))
                       :changed-ratio (it/mh-log-ratio old next (constantly 0.0) nil nil delta)
                       :result (:trace/result next) :source (:trace/result old)}))))]
    (is (contains? delta :nested-z))
    (is (contains? delta :nested-y))
    (is absent?)
    (is (< (Math/abs (- full-ratio changed-ratio)) 1e-10))
    (is (= [false] result))
    (is (= [true] source))))

(deftest gibbs-refuses-changing-latent-structure
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fixed site structure"
                        (execute branch-child
                                 (fn [t] (it/update-gibbs t :branch {:keep-old? true}))))))

(defn- coin []
  (spin (let [x (sample (d/flip 0.3) :id :x :init true)]
          (observe (d/flip (if x 0.9 0.2)) true :id :y)
          x)))

(deftest gibbs-rescores-ordinary-traces-at-the-requested-temperature
  (let [scores (execute coin
                        (fn [t] (pcps/async
                                 (:log-weights (pcps/await (it/update-gibbs t :x {:temperature 0.5 :keep-old? true}))))))
        expected (+ (Math/log (/ 0.3 0.7)) (* 0.5 (Math/log (/ 0.9 0.2))))]
    (is (< (Math/abs (- (- (get scores true) (get scores false)) expected)) 1e-10))))

(deftest an-unsupported-resumed-parent-is-refused-without-consuming-source
  (let [result (execute #(spin (let [xs (aw/await (keyed :items [0]
                                                         (fn [_] (spin (sample (d/flip 0.5) :id :x :init true)))))]
                                 (when (first xs) (observe (d/flip 0.8) true :id :tail))
                                 xs))
                        (fn [t]
                          (pcps/async
                           (let [error (try
                                         (pcps/await (gfi/update t {:x false}))
                                         nil
                                         (catch clojure.lang.ExceptionInfo e (:type (ex-data e))))]
                             {:error error :source (:trace/result t)
                              :tail? (contains? (:trace/entries t) :tail)}))))]
    (is (= ::it/no-replay-origin (:error result)))
    (is (= [true] (:source result)))
    (is (:tail? result))))

(deftest rewritten-likelihoods-replace-their-policy-ledger-contributions
  (let [rows (execute #(spin (aw/await (keyed :coins [0] (fn [_] (coin)))))
                      (fn [source]
                        (pcps/async
                         (loop [t source i 0 rows []]
                           (if (= i 6) rows
                               (let [x (get-in t [:trace/entries :x :value])
                                     next (pcps/await
                                           (it/reexecute t #{:x}
                                                         (it/policy {:keep? true
                                                                     :draw (fn [s _] (when (= :x (:savepoint/address s))
                                                                                       {:value (not x) :symmetric? true}))})
                                                         {:seed (+ 520 i)}))
                                     row [(not x) (get-in next [:trace/entries :x :value]) (it/log-weight next)
                                          (Math/log (if (not x) 0.9 0.2))]]
                                 (when-not (identical? t source) (trace/release! t next))
                                 (recur next (inc i) (conj rows row))))))))]
    (is (= 6 (count rows)))
    (doseq [[expected actual ledger correct] rows]
      (is (= expected actual))
      (is (< (Math/abs (- ledger correct)) 1e-10)))))

(deftest numerical-context-cannot-hide-a-tempered-target-in-policy-options
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"temperature one"
                        (transition/step (transition/prepare (k/hmc-kernel 1)) {}
                                         {:policy-options {:temperature 0.5}})))
  (is (= 0.5 (it/target-temperature {:policy-options {:temperature 0.5}})))
  (is (= 1.0 (it/target-temperature {:temperature 1.0 :policy-options {:temperature 0.5}}))))

(deftest cached-ordinary-descendants-cannot-silently-ignore-a-requested-update
  (let [result (execute
                #(spin (sample (d/flip 0.5) :id :prefix :init true)
                       (aw/await (keyed :outer [0]
                                        (fn [_]
                                          (spin (aw/await (spin (sample (d/flip 0.5) :id :x :init true))))))))
                (fn [t] (pcps/async
                         (let [next (:trace (pcps/await (gfi/update t {:x false})))]
                           [(:trace/result t) (:trace/result next) (get (it/choices next) :x)]))))]
    (is (= [[true] [false] false] result))))
