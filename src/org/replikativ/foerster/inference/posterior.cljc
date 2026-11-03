(ns org.replikativ.foerster.inference.posterior
  "Reading a posterior measure: weighted statistics of the program's value
  (`query`), draws passed to a function (`predict`), and posterior
  predictive replays of a model (`predictive`)."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as-alias core]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.runtime.gfi :as gfi]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]))

(defn query
  "Weighted statistics of a numeric function of the program's value over
  `measure`: `query-fn` is `identity` (the value itself), a keyword (a field
  of a map value) or a function of the value.

  Returns {:mean :variance :std-dev :quantiles :samples :weights :type};
  `:samples` and `:weights` are the particles' values and normalized
  weights."
  [measure query-fn]
  (let [extract-fn (cond
                     ;; identity means "get the main result"
                     (= query-fn identity) m/get-value
                     ;; keyword means "extract field from result"
                     (keyword? query-fn) (fn [ctx] (get (m/get-value ctx) query-fn))
                     ;; function: compose with get-value
                     :else (fn [ctx] (query-fn (m/get-value ctx))))]
    (m/measure-stats measure extract-fn)))

(defn predict
  "`num-samples` draws from `measure`, resampled by weight, each passed to
  `pred-fn` — which gets the particle (a `Sample`, or a context), so
  `m/get-value` reads its program value and `m/get-trace` its trace."
  [measure pred-fn num-samples]
  (let [samples (m/sample-measure measure num-samples)]
    (mapv (fn [[ctx _]] (pred-fn ctx)) samples)))

(defn predictive
  "`n` posterior predictive draws: particles of `measure` drawn by weight,
  each replayed through `model` with its latent choices held and its
  observed sites drawing fresh values instead of scoring the data. A prior
  predictive draw runs `model` with `(itrace/policy {:simulate-observed?
  true})` (`gfi/run-policy`); `gfi/simulate` would score the data instead.

  `model` need not be the program that was conditioned: the same program on
  new inputs (more weeks, other covariates) predicts them, its latent sites
  held where the posterior has them and drawn from their prior where it
  does not (a new group's effect). `:interventions` ({selector transform},
  as `foerster.runtime.trace/policy` takes them) sets sites by the do-operator
  first — a scenario under the posterior.

  Returns a spin resolving a vector of {:value v :observations {address x}}:
  the program's value and what each observed site drew."
  [model measure n & [{:keys [interventions]}]]
  (spin
   (loop [[[particle _] & more] (m/sample-measure measure n) out []]
     (if-not particle
       out
       (let [_ (when (nil? (m/get-trace particle))
                 (throw (ex-info "Posterior predictive replay needs traces; use amortized/predictive with an explicit forward binding for learned draws"
                                 {:type ::core/missing-predictive-trace})))
             latents (into {} (keep (fn [[address {:keys [value observed? deterministic?]}]]
                                      (when-not (or observed? deterministic?) [address value])))
                           (m/get-trace particle))
             t (await (gfi/run-policy model (itrace/policy {:constraints latents
                                                            :interventions interventions
                                                            :simulate-observed? true})))
             drawn (into {} (mapcat (fn [[address {:keys [value note]}]]
                                      (cond
                                        (:simulated? note) [[address value]]
                                        ;; a block's own observations, when it can draw them
                                        (block/block-dist? (:dist note))
                                        (when-let [simulate (block/capability (:block (:dist note)) :simulate)]
                                          (map (fn [[k v]] [[address k] v])
                                               (simulate (double-array value) (:inputs (:dist note))))))))
                         (:trace/entries t))]
         (await (gfi/close! t))
         (recur more (conj out {:value (:trace/result t) :observations drawn})))))))
