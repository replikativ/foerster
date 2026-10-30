(ns org.replikativ.foerster.program-test
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.trace-owner :as owner]
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
        (is (= 128 (count (m/get-particles result))))
        (is (< (Math/abs (- (Math/log 0.41) (:log-mass result))) 0.15))
        (is (< (Math/abs (- (:log-mass result) (m/log-marginal result))) 1e-10))
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
    (is (= 256 (count (m/get-particles result))))
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
           (mapv (comp m/get-value first) (m/get-particles result))))))

(deftest unsupported-options-and-unscored-proposals-fail-before-running
  (is (thrown? clojure.lang.ExceptionInfo (ip/run {:world-opts {:systems :all}} identity)))
  (is (thrown? clojure.lang.ExceptionInfo (ip/trace-target nil {:world-opts {:systems :all}})))
  (is (thrown? clojure.lang.ExceptionInfo (ip/value-target identity identity nil))))

(deftest long-moves-reclaim-intermediate-traces-before-completion
  (let [reclaimed (atom [])
        reclaim! owner/reclaim!]
    (with-redefs [owner/reclaim! (fn [dropped live]
                                   (swap! reclaimed conj {:dropped (count dropped) :live (count live)})
                                   (reclaim! dropped live))]
      (let [result (run #(ip/run {:seed 42}
                                 (fn [scope]
                                   (spin
                                    (let [p (await (ip/initialize scope (ip/trace-target (coin-model) {:beta 1.0}) 4))
                                          p (await (ip/move scope p (k/cycle 2 [(k/single-site-mh-kernel 10)])))]
                                      (ip/finish scope p))))))]
        (is (= 4 (count (m/get-particles result))))
        ;; Every completed inner move offers its displaced state immediately;
        ;; retaining all 80 visited traces until the outer move ends would leak
        ;; worlds for the duration of a long programmable transition.
        (is (>= (count @reclaimed) 81))
        (is (every? #(= 1 (:dropped %)) (take 80 @reclaimed)))
        (is (every? #(= 4 (:live %)) @reclaimed))))))

(deftest trace-forks-survive-source-retirement-and-stale-handles-fail
  (let [stale (atom nil)
        result (run #(ip/run {:seed 31}
                             (fn [scope]
                               (spin
                                (let [source (await (ip/initialize scope (ip/trace-target (coin-model)) 6))
                                      child (await (ip/fork scope source))
                                      parent (ip/advance scope source (ip/temperature (:target source) 1.0))
                                      _ (reset! stale (try (ip/finish scope source) nil
                                                           (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))
                                      _ (ip/finish scope parent)
                                      child (await (ip/resample scope child))
                                      child (ip/advance scope child (ip/temperature (:target child) 1.0))
                                      child (await (ip/move scope child (k/single-site-mh-kernel 4)))]
                                  (ip/finish scope child))))))]
    (is (= :org.replikativ.foerster.program/stale-population @stale))
    (is (= 6 (count (m/get-particles result))))
    (is (every? boolean? (map (comp m/get-value first) (m/get-particles result))))
    (is (Double/isFinite (m/log-marginal result)))))

(deftest population-moves-reject-nested-warmup
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"frozen tuning"
                        (run #(ip/run (fn [scope]
                                        (spin
                                         (let [p (await (ip/initialize scope (ip/trace-target (coin-model) {:beta 1.0}) 2))
                                               p (await (ip/move scope p
                                                                 (k/cycle 1 [(k/single-site-mh-kernel 1)
                                                                             (k/random-walk-mh-kernel 2 {:burn 1})])))]
                                           (ip/finish scope p)))))))))

(deftest transport-requires-resampling-away-zero-weight-slots
  (let [draws (atom false)
        source (ip/value-target #(if % 0.0 ##-Inf) #(swap! draws not) (constantly (- (Math/log 2.0))))
        destination (ip/retarget source :next (constantly 0.0))
        kernel (ip/transport-kernel source destination
                                    (fn [x] {:value x :log-q 0.0 :reverse nil})
                                    (fn [_ _] 0.0))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"resample first"
                          (run #(ip/run (fn [scope]
                                          (spin
                                           (let [p (await (ip/initialize scope source 2))
                                                 p (await (ip/transport scope p destination kernel))]
                                             (ip/finish scope p))))))))))

(deftest live-handles-cannot-escape-their-resource-scope
  (let [escaped (atom nil)]
    (run #(ip/run (fn [scope]
                    (spin
                     (let [p (await (ip/initialize scope (ip/trace-target (coin-model)) 2))]
                       (reset! escaped [scope p])
                       (ip/finish scope p))))))
    (let [[scope p] @escaped]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no longer live"
                            (ip/finish scope p))))))

(deftest rejected-transport-preconditions-leave-trace-populations-usable
  (let [rejected (atom nil)
        result (run #(ip/run {:seed 42}
                             (fn [scope]
                               (spin
                                (let [model (spin (let [h (sample (d/flip 0.5) :id :h)]
                                                    (observe (d/flip (if h 1.0 0.0)) true :id :y)
                                                    h))
                                      source (ip/trace-target model {:beta 1.0})
                                      destination (ip/temperature source 1.0)
                                      kernel (ip/transport-kernel source destination
                                                                  (fn [_ _] {:updates {} :log-q 0.0 :reverse nil})
                                                                  (fn [_ _] 0.0))
                                      p (await (ip/initialize scope source 16))
                                      _ (try (await (ip/transport scope p destination kernel))
                                             (catch clojure.lang.ExceptionInfo e
                                               (reset! rejected (:type (ex-data e)))))
                                      p (await (ip/resample scope p))]
                                  (ip/finish scope p))))))]
    (is (= :org.replikativ.foerster.program/zero-weight-transport @rejected))
    (is (every? true? (map (comp m/get-value first) (m/get-particles result))))))
