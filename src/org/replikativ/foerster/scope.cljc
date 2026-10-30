(ns org.replikativ.foerster.scope
  "An asynchronous lifetime for inference resources. Acquisitions register
  before execution starts; close joins acquisitions and attempts every release."
  (:refer-clojure :exclude [await])
  (:require [is.simm.partial-cps.async :as pcps]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defn create []
  (atom {:phase :open :acquiring 0 :resources [] :waiters []}))

(defn- publish-acquisition! [scope entry]
  (let [[_ after] (swap-vals! scope
                              (fn [s]
                                (cond-> (update s :acquiring dec)
                                  entry (update :resources conj entry))))]
    (when (and (= :closing (:phase after)) (zero? (:acquiring after)))
      (when-let [ready (:ready after)] (ready)))))

(defn acquire!
  "Allocate synchronously and register its awaitable release before returning.
  Release receives the resource and returns a CPS operation. Close waits for
  acquisitions already started; new acquisitions after closing are refused."
  [scope allocate release]
  (swap! scope (fn [s]
                 (when-not (= :open (:phase s))
                   (throw (ex-info "Inference scope is closed" {:type ::closed})))
                 (update s :acquiring inc)))
  (let [resource (try (allocate)
                      (catch #?(:clj Throwable :cljs :default) e
                        (publish-acquisition! scope nil)
                        (throw e)))]
    (publish-acquisition! scope #(release resource))
    resource))

(defn- release-resources [releases]
  ;; Partial-CPS supplies a stack-safe trampoline without allocating a Spin
  ;; world for each resource. Downstream callback execution is not inside a
  ;; manual try/catch that could mistake stack overflow for a release failure.
  (pcps/async
   (loop [remaining (seq releases) errors []]
     (if-not remaining
       errors
       (let [error (try (pcps/await ((first remaining))) nil
                        (catch #?(:clj Throwable :cljs :default) e e))]
         (recur (next remaining) (cond-> errors error (conj error))))))))

(defn close!
  "CPS finalizer. Repeated/concurrent closes await the same outcome. Releases
  run in reverse acquisition order, including resources acquired while closing."
  [scope]
  (fn [resolve reject]
    (letfn [(deliver! [outcome]
              (let [[before _] (swap-vals! scope assoc :phase :closed :outcome outcome
                                           :resources [] :waiters [] :ready nil)]
                (doseq [[res rej] (:waiters before)]
                  (if-let [error (:error outcome)] (rej error) (res nil)))))
            (ready! []
              (let [[before _] (swap-vals! scope
                                           (fn [s] (if (and (= :closing (:phase s))
                                                            (zero? (:acquiring s)))
                                                     (assoc s :phase :releasing :ready nil) s)))]
                (when (and (= :closing (:phase before)) (zero? (:acquiring before)))
                  ((release-resources (reverse (:resources before)))
                   (fn [errors]
                     (deliver! (when (seq errors)
                                 {:error (ex-info "Inference resource cleanup failed"
                                                  {:type ::cleanup-failed :errors errors}
                                                  (first errors))})))
                   (fn [e] (deliver! {:error e}))))))]
      (let [[before after] (swap-vals! scope
                                       (fn [s]
                                         (cond-> s
                                           (not= :closed (:phase s)) (update :waiters conj [resolve reject])
                                           (= :open (:phase s)) (assoc :phase :closing :ready ready!))))]
        (cond
          (= :closed (:phase before))
          (if-let [error (get-in before [:outcome :error])] (reject error) (resolve nil))
          (and (= :closing (:phase after)) (zero? (:acquiring after))) (ready!))))))

(defn run
  "Return a Spin executing (body scope), which returns a Spin or CPS operation.
  Cancellation, failure and success all finalize owned resources before exit."
  [body]
  (spin-core/with-causal-descendant-egress
    (spin
     (let [scope (create)
           failed (volatile! nil)]
       (try
         (await (body scope))
         (catch #?(:clj Throwable :cljs :default) e
           (vreset! failed e)
           (throw e))
         (finally
           (try
             (await-finalization (close! scope))
             (catch #?(:clj Throwable :cljs :default) cleanup
               (if-let [original @failed]
                 (throw (ex-info "Inference failed and cleanup also failed"
                                 {:type ::cleanup-failed :cleanup cleanup} original))
                 (throw cleanup))))))))))
