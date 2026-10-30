(ns org.replikativ.foerster.cursor-test
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is]])
            [org.replikativ.foerster.cursor :as cursor]
            [is.simm.partial-cps.async :as pcps])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defn- start! [operation]
  (let [outcome (atom ::pending)]
    (operation #(reset! outcome [:ok %]) #(reset! outcome [:error %]))
    outcome))

(defn- value-of [outcome]
  (let [[status value] @outcome]
    (is (= :ok status) (str @outcome))
    value))

(defn- raw-step [history closed]
  {:history history
   :closed closed
   :push (fn [y] (fn [resolve _] (resolve (raw-step (conj history y) closed))))
   :fork (fn [] (fn [resolve _] (resolve (raw-step history (atom 0)))))
   :close-operation (fn [resolve _] (swap! closed inc) (resolve nil))})

(deftest consumed-and-closed-steps-cannot-push-again
  (let [closed (atom 0)
        first-step (cursor/guard (raw-step [] closed))
        next-step (value-of (start! ((:push first-step) :first)))
        stale (start! ((:push first-step) :stale))]
    (is (= [:first] (:history next-step)))
    (is (= :error (first @stale)))
    (is (= ::cursor/stale-step (:type (ex-data (second @stale)))))
    (value-of (start! (:close-operation next-step)))
    (let [after-close (start! ((:push next-step) :closed))]
      (is (= :error (first @after-close)))
      (is (= :closed (:phase (ex-data (second @after-close))))))
    (is (= 1 @closed))))

(deftest an-in-flight-push-cannot-lose-its-waiter
  (let [pending (atom nil)
        calls (atom [])
        first-step (cursor/guard
                    {:push (fn [y]
                             (fn [resolve _]
                               (swap! calls conj y)
                               (reset! pending resolve)))
                     :close-operation (fn [resolve _] (resolve nil))})
        first-result (start! ((:push first-step) :first))
        conflicting (start! ((:push first-step) :second))]
    (is (= ::pending @first-result))
    (is (= :error (first @conflicting)))
    (is (= :pushing (:phase (ex-data (second @conflicting)))))
    (is (= [:first] @calls) "the second call never reaches the controller")
    (@pending (raw-step [:first] (atom 0)))
    (let [next-step (value-of first-result)]
      (is (= [:first] (:history next-step)))
      (value-of (start! (:close-operation next-step))))))

(deftest a-fork-has-its-own-generation-and-lifetime
  (let [parent (cursor/guard (raw-step [] (atom 0)))
        child (value-of (start! ((:fork parent))))
        parent-next (value-of (start! ((:push parent) :parent)))
        child-next (value-of (start! ((:push child) :child)))]
    (is (= [:parent] (:history parent-next)))
    (is (= [:child] (:history child-next)))
    (value-of (start! (:close-operation parent-next)))
    (is (= 1 @(:closed parent-next)))
    (is (zero? @(:closed child-next)))
    (let [child-final (value-of (start! ((:push child-next) :still-open)))]
      (is (= [:child :still-open] (:history child-final)))
      (value-of (start! (:close-operation child-final)))
      (is (= 1 @(:closed child-final))))))

(deftest close-is-awaited-and-idempotent-across-step-versions
  (let [finish-close (atom nil)
        closes (atom 0)
        close-operation (fn [resolve _]
                          (swap! closes inc)
                          (reset! finish-close resolve))
        step (cursor/guard {:push (fn [_]
                                    (fn [resolve _]
                                      (resolve {:close-operation close-operation})))
                            :close-operation close-operation})
        next-step (value-of (start! ((:push step) :next)))
        first-close (start! (:close-operation step))
        second-close (start! (:close-operation next-step))]
    (is (= 1 @closes))
    (is (= ::pending @first-close))
    (is (= ::pending @second-close))
    (@finish-close nil)
    (is (= [:ok nil] @first-close @second-close))
    (is (= [:ok nil] @(start! (:close-operation step))))
    (is (= 1 @closes))))

(deftest fire-and-forget-close-starts-cleanup-inside-a-trampoline
  (let [closed (atom 0)
        finish-close (atom nil)
        step (cursor/guard {:close-operation (fn [resolve _]
                                               (swap! closed inc)
                                               (reset! finish-close resolve))})
        invoked (start! (pcps/async ((:close step)) :invoked))
        joined (start! (:close-operation step))]
    (is (= [:ok :invoked] @invoked))
    (is (= 1 @closed) "raw cleanup starts even when the legacy close return is ignored")
    (is (= ::pending @joined))
    (when-let [done @finish-close] (done nil))
    (is (= [:ok nil] @joined))
    (is (= [:ok nil] @(start! (:close-operation step))))
    (is (= 1 @closed))))
