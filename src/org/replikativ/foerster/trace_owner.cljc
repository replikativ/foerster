(ns org.replikativ.foerster.trace-owner
  "Reachability reclamation for complete trace populations sharing anchors."
  (:require [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.engine.protocols :as rtp]))

(defn- anchor-ids [traces]
  (into #{} (comp (mapcat (comp vals :trace/entries))
                  (keep :savepoint)
                  (map trace/anchor-id))
        traces))

(defn reclaim!
  "Give back what `dropped` traces hold that no `live` trace refers to."
  [dropped live]
  (let [live-anchors (anchor-ids live)
        live-worlds (into #{} (map (comp :fork-id :trace/world)) live)]
    (doseq [t (vals (into {} (map (juxt (comp :fork-id :trace/world) identity)) dropped))]
      (doseq [a (keep :savepoint (vals (:trace/entries t)))
              :when (and (not (contains? live-anchors (trace/anchor-id a)))
                         (sp/pending? a))]
        (try (sp/abandon a) (catch #?(:clj Throwable :cljs :default) _ nil)))
      (let [w (:trace/world t)]
        (when-not (contains? live-worlds (:fork-id w))
          (sp/release-world! (:trace/session t) w))))))

(defn reclaim
  "CPS population ownership phase. Complete projections no longer need the
  old engine reuse source; clear that edge before releasing unreachable worlds."
  [old seen live]
  (fn [resolve reject]
    (try
      (doseq [t live]
        (rtp/swap-state! (:trace/world t) [:engine/reuse-source] (constantly nil)))
      (reclaim! (into (vec old) seen) live)
      (resolve nil)
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))
