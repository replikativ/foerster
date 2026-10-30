(ns org.replikativ.foerster.cursor
  "Versioned inference cursors: pushes and forks cannot overwrite a waiter or
  consume a later observation through an old step. Close is controller-wide."
  (:require [is.simm.partial-cps.async :as pcps]
            [is.simm.partial-cps.runtime :as cps-runtime])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defn guard [initial]
  (let [lease (atom {:phase :idle :generation 0 :step initial :closing []})]
    (letfn [(close-operation [resolve reject]
              (let [[before _] (swap-vals! lease
                                           (fn [s]
                                             (if (= :closed (:phase s)) s
                                                 (-> s (assoc :phase :closing)
                                                     (update :closing conj [resolve reject])))))]
                (cond
                  (= :closed (:phase before))
                  (if-let [e (:error before)] (reject e) (resolve nil))
                  (= :closing (:phase before)) nil
                  :else
                  (let [done (fn [error]
                               (let [[old _] (swap-vals! lease assoc :phase :closed :error error
                                                         :step nil :closing [])]
                                 (doseq [[res rej] (:closing old)] (if error (rej error) (res nil)))))
                        raw (:step before)]
                    (loop [operation ((pcps/async
                                       (try
                                         (if-let [close (:close-operation raw)]
                                           (pcps/await close)
                                           ((:close raw)))
                                         nil
                                         (catch #?(:clj Throwable :cljs :default) e e)))
                                      done done)]
                      (when (cps-runtime/thunk? operation)
                        (recur (cps-runtime/force-thunk operation))))))))
            (claim! [generation phase]
              (swap! lease
                     (fn [s]
                       (when-not (and (= :idle (:phase s)) (= generation (:generation s)))
                         (throw (ex-info "Inference cursor step is stale, closed or already in flight"
                                         {:type ::stale-step :generation generation
                                          :current (:generation s) :phase (:phase s)})))
                       (assoc s :phase phase))))
            (perform [generation mode start]
              (pcps/async
               (claim! generation mode)
               (try
                 (let [raw (pcps/await (start))
                       [before _] (swap-vals! lease
                                              (fn [s]
                                                (if (= mode (:phase s))
                                                  (cond-> (assoc s :phase :idle)
                                                    (= mode :pushing) (assoc :generation (inc generation) :step raw)) s)))]
                   (if (= mode (:phase before))
                     (if (= mode :pushing) (wrap-step raw (inc generation)) (guard raw))
                     (do
                       (when (= mode :forking)
                         (if-let [close (:close-operation raw)] (pcps/await close) ((:close raw))))
                       (throw (ex-info "Inference cursor closed during operation" {:type ::closed})))))
                 (catch #?(:clj Throwable :cljs :default) e
                   (swap! lease #(if (= mode (:phase %))
                                   (assoc % :phase (if (= mode :forking) :idle :failed)) %))
                   (throw e)))))
            (wrap-step [step generation]
              (cond-> (assoc step :close (fn [] (close-operation (fn [_]) (fn [_])))
                             :close-operation close-operation)
                (:push step) (assoc :push (fn [y] (perform generation :pushing #((:push step) y))))
                (:fork step) (assoc :fork (fn [] (perform generation :forking #((:fork step)))))))]
      (wrap-step initial 0))))

(defn operation [start]
  (pcps/async (guard (pcps/await start))))
