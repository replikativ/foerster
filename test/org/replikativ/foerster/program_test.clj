(ns org.replikativ.foerster.program-test
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]))

(defn- run [make]
  (binding [ec/*execution-context* (ctx/create-execution-context)]
    (let [p (promise)]
      ((make) #(deliver p [:ok %]) #(deliver p [:error %]))
      (let [[status result :as outcome] (deref p 30000 [::timeout nil])]
        (if (= :ok status) result
            (throw (if (= :error status) result (ex-info "Program timed out" {:outcome outcome}))))))))

(defn- coin-model []
  (spin (let [h (sample (d/flip 0.3) :id :h)]
          (observe (d/flip (if h 0.9 0.2)) true :id :y)
          h)))

(deftest a-custom-trace-resample-move-program-releases-every-session
  (let [closed (atom 0) close! sp/close!]
    (with-redefs [sp/close! (fn [s] (fn [resolve reject]
                                      ((close! s) (fn [v] (swap! closed inc) (resolve v)) reject)))]
      (let [result (run #(ip/run {:seed 42}
                                 (fn [scope]
                                   (spin
                                    (let [p (await (ip/initialize scope (ip/trace-target (coin-model)) 128))
                                          p (ip/advance scope p (ip/temperature (:target p) 1.0))
                                          p (await (ip/resample scope p))
                                          p (await (ip/move scope p (k/single-site-mh-kernel 3)))]
                                      (ip/finish scope p))))))]
        (is (= 128 @closed))
        (is (= 128 (count (m/get-particles (:measure result)))))
        (is (< (Math/abs (- (Math/log 0.41) (:log-mass result))) 0.15))
        (is (< (Math/abs (- (:log-mass result) (m/log-marginal (:measure result)))) 1e-10))
        (is (= [:advance :resample :move] (mapv :operation (:history result))))
        (is (= :unverified (get-in result [:inference-contract :normalizer :status])))))))

(deftest value-and-trace-programs-share-the-mass-convention
  (let [prior (d/flip 0.3)
        result (run #(ip/run {:seed 7}
                             (fn [scope]
                               (spin
                                (let [initial (ip/value-target (fn [x] (d/logpdf prior x))
                                                               (fn [] (d/draw prior)) (fn [x] (d/logpdf prior x)))
                                      p (await (ip/initialize scope initial 512))
                                      next (ip/retarget initial :observed
                                                        (fn [x] (+ (d/logpdf prior x) (Math/log (if x 0.9 0.2)))))
                                      p (ip/advance scope p next)
                                      p (await (ip/resample scope p {:particles 256}))
                                      exact (ip/invariant-kernel next (fn [_] (d/draw (d/flip (/ 0.27 0.41)))))
                                      p (await (ip/move scope p exact))]
                                  (ip/finish scope p))))))]
    (is (= 256 (count (m/get-particles (:measure result)))))
    (is (< (Math/abs (- (Math/log 0.41) (:log-mass result))) 0.15))
    (is (= :unverified (get-in result [:inference-contract :normalizer :status])))))

(deftest unfinished-programs-and-incompatible-targets-are-refused
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Finish the population"
                        (run #(ip/run (fn [scope] (ip/initialize scope (ip/trace-target (coin-model)) 2))))))
  (is (thrown? clojure.lang.ExceptionInfo (ip/trace-target nil {:world-policy :fork})))
  (is (thrown? clojure.lang.ExceptionInfo
               (ip/advance {:owner :a} {:owner :b} {}))))

(deftest markov-output-is-not-an-evidence-provider
  (let [measure (run #(infer/kernel-infer (coin-model) (k/single-site-mh-kernel 2) 2))]
    (is (= :markov-chain (get-in measure [:inference-contract :semantics])))
    (is (nil? (m/log-marginal measure)))
    (is (re-find #"unavailable" (pr-str measure)))))

(deftest nested-conditionals-carry-their-approximation-provenance
  (let [exact (run #(infer/conditional (coin-model) {:method :enumerate}))
        approximate (run #(infer/conditional (coin-model) {:method :smc :particles 16 :seed 42}))]
    (is (true? (get-in (meta exact) [:inference/nesting :exact?])))
    (is (false? (get-in (meta approximate) [:inference/nesting :exact?])))
    (is (= 16 (get-in (meta approximate) [:inference/nesting :particles])))))

(deftest different-scores-at-one-position-have-distinct-target-identities
  (let [prior (d/flip 0.3)
        target (ip/value-target (fn [x] (d/logpdf prior x))
                                (fn [] (d/draw prior)) (fn [x] (d/logpdf prior x)))
        a (ip/retarget target :same-position (constantly 0.0))
        b (ip/retarget target :same-position (constantly -1.0))
        kernel (ip/invariant-kernel a (fn [x] x))]
    (is (not= (:id a) (:id b)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must target this population"
                          (run #(ip/run (fn [scope]
                                          (spin
                                           (let [p (await (ip/initialize scope target 2))
                                                 p (ip/advance scope p b)
                                                 p (await (ip/move scope p kernel))]
                                             (ip/finish scope p))))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"value representation"
                          (ip/retarget (ip/trace-target nil) :custom-score (constantly 0.0))))))

(deftest repeated-initialization-does-not-reuse-the-same-random-stream
  (let [normal (d/normal 0.0 1.0)
        target (ip/value-target (fn [x] (d/logpdf normal x))
                                (fn [] (d/draw normal)) (fn [x] (d/logpdf normal x)))
        snapshots (atom nil)
        execute (fn []
                  (run #(ip/run {:seed 42}
                                (fn [scope]
                                  (spin
                                   (let [a (await (ip/initialize scope target 8))
                                         b (await (ip/initialize scope target 8))]
                                     (reset! snapshots [(:particles a) (:particles b)])
                                     (ip/finish scope b))))))
                  @snapshots)
        first-run (execute)
        second-run (execute)]
    (is (not= (first first-run) (second first-run)))
    (is (= first-run second-run))))

(deftest mutable-value-transitions-copy-before-modifying-their-input
  (let [normal (d/normal 0.0 1.0)
        source (atom nil) before (atom nil)
        target (ip/value-target (fn [x] (d/logpdf normal (aget ^doubles x 0)))
                                (fn [] (double-array [(d/draw normal)]))
                                (fn [x] (d/logpdf normal (aget ^doubles x 0)))
                                {:copy aclone :project (fn [x] (aget ^doubles x 0))})
        kernel (ip/invariant-kernel target (fn [x] (aset-double x 0 (- (aget ^doubles x 0))) x))
        result (run #(ip/run {:seed 42}
                             (fn [scope]
                               (spin
                                (let [p (await (ip/initialize scope target 4))
                                      _ (reset! source (:particles p))
                                      _ (reset! before (mapv vec (:particles p)))
                                      p (await (ip/resample scope p))
                                      p (await (ip/move scope p kernel))]
                                  (ip/finish scope p))))))]
    (is (= @before (mapv vec @source)))
    (is (= (mapv (comp - first) @before)
           (mapv (comp m/get-value first) (m/get-particles (:measure result)))))))

(deftest unsupported-options-and-unscored-proposals-fail-before-running
  (is (thrown? clojure.lang.ExceptionInfo (ip/run {:world-opts {:systems :all}} identity)))
  (is (thrown? clojure.lang.ExceptionInfo (ip/trace-target nil {:world-opts {:systems :all}})))
  (is (thrown? clojure.lang.ExceptionInfo (ip/value-target identity identity nil))))
