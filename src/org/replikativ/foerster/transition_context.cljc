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
  (when (not= 1.0 (double (it/target-temperature context)))
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
  (it/tempered-score (it/target-temperature context)))

(defn score [context t]
  (transduce (map (entry-score context)) + 0.0 (it/entries t)))

(defn changed-outside
  "The addresses, other than `address`, whose score differs between traces
  `old` and `new` (or that one of them lacks): what a block's write-back
  changed outside the block."
  [context old new address]
  (let [scores (fn [t] (into {} (map (juxt :address (entry-score context))) (it/entries t)))
        a (scores old) b (scores new)]
    (vec (for [k (distinct (concat (clojure.core/keys a) (clojure.core/keys b)))
               :when (and (not= k address)
                          (let [x (get a k) y (get b k)]
                            (or (nil? x) (nil? y) (> (Math/abs (double (- x y))) 1e-9))))]
           k))))

(defn incomplete-target
  "The error a block move that needs its block's complete conditional
  raises when its write-back changed the score of sites outside the block
  by `outside` in total."
  [kind error-type context old new address outside]
  (let [changed (changed-outside context old new address)]
    (ex-info (str kind " needs the block's target to be the complete conditional of its latents, "
                  "but moving block " (pr-str address) " changed the log density of "
                  (pr-str (vec (take 5 changed))) (when (> (count changed) 5) " …")
                  " outside it (by " outside "). " kind " follows the block's own density only, so "
                  "every factor its latents touch must be inside the block and no other site may "
                  "depend on them. Either put the dependent latents into the same block (a "
                  "hierarchy as one block site), or move this block with HMC (k/hmc-kernel) or "
                  "delayed acceptance (k/delayed-acceptance-kernel), which accept on the whole "
                  "trace and allow such blocks.")
             {:type error-type :address address :outside outside :changed changed})))

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
