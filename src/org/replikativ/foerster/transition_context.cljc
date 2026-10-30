(ns org.replikativ.foerster.transition-context
  "Target, replay and ownership context shared by numerical and trace moves."
  (:refer-clojure :exclude [keys])
  (:require [org.replikativ.foerster.trace :as it]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.effects.savepoint :as sp]))

(def keys
  #{:temperature :policy-options :constraints :until :anchor?
    :keep-old? :shared-anchors? :on-visit})

(defn require-untempered! [context family]
  (when (and (some? (:temperature context)) (not= 1.0 (double (:temperature context))))
    (throw (ex-info "This numerical transition currently requires temperature one"
                    {:type ::unsupported-temperature :kernel family :temperature (:temperature context)}))))

(defn policy [context draw]
  (it/policy (cond-> (assoc (:policy-options context)
                            :keep? true
                            :constraints (merge (get-in context [:policy-options :constraints]) (:constraints context))
                            :draw draw)
               (some? (:temperature context)) (assoc :temperature (:temperature context)))))

(defn replay-options
  ([context]
   (cond-> {:anchor? (or (:anchor? context) it/anchor?)}
     (:until context) (assoc :until (:until context))))
  ([context old move-key]
   (cond-> (replay-options context)
     (sp/seed (:trace/world old))
     (assoc :seed (sp/derive-seed (sp/seed (:trace/world old)) ::replay move-key)))))

(defn at-frontier? [context candidate]
  (or (not (:until context)) (some? (:trace/pending candidate))))

(defn entry-score [context]
  ;; Old entries may have been recorded under an earlier SMC temperature.
  (it/tempered-score (get context :temperature 1.0)))

(defn score [context t]
  (transduce (map (entry-score context)) + 0.0 (it/entries t)))

(defn settle!
  "Release the rejected proposal, or the displaced state according to its
  owner's retention policy. A population owner reclaims shared anchors later."
  [context old candidate accepted?]
  (cond
    (not accepted?) (trace/release! candidate old)
    (:keep-old? context) nil
    (:shared-anchors? context)
    (do (when-let [pending (:trace/pending old)]
          (when (sp/pending? pending) (sp/abandon pending)))
        (sp/release-world! (:trace/session old) (:trace/world old)))
    :else (trace/release! old candidate)))
