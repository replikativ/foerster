(ns org.replikativ.foerster.runtime.trace-owner
  "Reachability reclamation for complete trace populations sharing anchors."
  (:require [org.replikativ.spindel.v2.savepoint :as sp]
            [org.replikativ.spindel.v2.trace :as trace]))

(defn- anchor-ids [traces]
  (into #{} (comp (mapcat (comp vals :trace/entries))
                  (keep :savepoint)
                  (map trace/anchor-id))
        traces))

(defn reclaim!
  "Give back what `dropped` traces hold that no `live` trace refers to: their
  anchors are disposed, their worlds released."
  [dropped live]
  (let [live-anchors (anchor-ids live)
        live-worlds (into #{} (keep :trace/world) live)]
    (doseq [t (vals (into {} (map (juxt :trace/world identity)) dropped))]
      (doseq [a (keep :savepoint (vals (:trace/entries t)))
              :when (not (contains? live-anchors (trace/anchor-id a)))]
        (try (sp/dispose a) (catch #?(:clj Throwable :cljs :default) _ nil)))
      (let [w (:trace/world t) s (:trace/session t)]
        (when (and w s (not (contains? live-worlds w)) (contains? @(:worlds s) w))
          (sp/release-world! s w))))))

(defn reclaim
  "CPS population ownership phase: reclaim what `old` and `seen` hold that
  `live` does not."
  [old seen live]
  (fn [resolve reject]
    (try
      (reclaim! (into (vec old) seen) live)
      (resolve nil)
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))
