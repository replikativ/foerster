(ns org.replikativ.foerster.arviz
  "A Markov-chain measure as ArviZ's InferenceData groups, for
  `az.from_dict`:

    (spit \"run.json\" (arviz/->json (arviz/inference-data chains {:predictive pp})))

    # python
    d = json.load(open(\"run.json\"))
    idata = az.from_dict(**d)

  Groups: `posterior` (the latent sites, and a block site's latents by name
  in their own coordinates), `log_likelihood` (`pointwise-log-likelihood`),
  `observed_data`, and with `:predictive` (the draws of
  `infer/predictive`) `posterior_predictive`. Each variable is nested
  [chain][draw] (the measure's `:chain-lengths`; chains of unequal length
  are cut to the shortest), then its own dimensions. Addresses become
  variable names: `:mu` is `mu`; `[:y 3]` is element 3 of an array `y`
  (several integer indices, a multidimensional one); any other address its
  printed form. A missing element is NaN."
  (:require [clojure.string :as str]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.measure :as m]))

(defn- var-of
  "[name index] of an address: index nil for a scalar variable."
  [address]
  (cond
    (keyword? address) [(name address) nil]
    (and (vector? address) (keyword? (first address)) (next address)
         (every? integer? (rest address)))
    [(name (first address)) (vec (rest address))]
    :else [(pr-str address) nil]))

(defn- nested
  "The values at `indices` ({index-vector value}) as nested vectors, NaN
  where an index is missing."
  [indices]
  (let [dims (apply mapv (fn [& is] (inc (apply max is))) (keys indices))]
    (letfn [(build [prefix [d & more]]
              (vec (for [i (range d)]
                     (let [p (conj prefix i)]
                       (if (seq more) (build p more) (get indices p ##NaN))))))]
      (build [] dims))))

(defn- variables
  "One draw's {address value} as {name value-or-nested}."
  [draw]
  (let [grouped (reduce (fn [acc [address v]]
                          (let [[nm idx] (var-of address)]
                            (if idx
                              (assoc-in acc [nm ::indices idx] v)
                              (assoc acc nm v))))
                        {} draw)]
    (into {} (map (fn [[nm v]] [nm (if (and (map? v) (::indices v)) (nested (::indices v)) v)]))
          grouped)))

(defn- by-chain
  "Per-draw variable maps as {name [chain][draw]…}, chains cut to the
  shortest."
  [draws chain-lengths]
  (let [lengths (or chain-lengths [(count draws)])
        n (apply min lengths)
        chains (loop [[l & more] lengths ds draws out []]
                 (if l (recur more (drop l ds) (conj out (vec (take n ds)))) out))
        names (distinct (mapcat keys draws))]
    (into {} (for [nm names]
               [nm (mapv (fn [chain] (mapv #(get % nm ##NaN) chain)) chains)]))))

(defn- latent-draw
  "The latent values of a particle's trace, a block site's latents by name."
  [trace]
  (into {} (mapcat (fn [[address {:keys [value observed? distribution]}]]
                     (cond
                       observed? nil
                       (block/block-dist? distribution)
                       (let [b (:block distribution)
                             x (block/constrain b value)
                             [nm idx] (var-of address)]
                         (for [{lname :name} (:block/latents (:description b))]
                           [(keyword (str (if idx (pr-str address) nm) "." (name lname)))
                            (block/latent b x lname)]))
                       (or (number? value) (and (vector? value) (every? number? value)))
                       [[address value]]))
                   trace)))

(defn inference-data
  "The InferenceData groups of `measure` (equally weighted draws, by
  chain), see the namespace. `:predictive`: draws of `infer/predictive`."
  [measure & [{:keys [predictive]}]]
  (let [particles (m/get-particles measure)
        traces (map (comp m/get-trace first) particles)
        lengths (:chain-lengths measure)
        observed (into {} (keep (fn [[address {:keys [observed? value]}]] (when observed? [address value])))
                       (first traces))]
    (cond-> {"posterior" (by-chain (mapv (comp variables latent-draw) traces) lengths)
             "log_likelihood" (by-chain (mapv variables (d/pointwise-log-likelihood measure)) lengths)
             "observed_data" (variables observed)}
      predictive (assoc "posterior_predictive"
                        (by-chain (mapv (comp variables :observations) predictive) nil)))))

(defn- json-number [x]
  (cond (NaN? x) "NaN"
        (= x ##Inf) "Infinity"
        (= x ##-Inf) "-Infinity"
        (integer? x) (str x)
        :else (str (double x))))

(defn ->json
  "`data` (maps, vectors, numbers, strings, keywords, booleans, nil) as JSON;
  NaN and infinities as Python's json module reads them."
  [data]
  (cond
    (map? data) (str "{" (str/join "," (map (fn [[k v]] (str (->json (if (keyword? k) (name k) (str k))) ":" (->json v))) data)) "}")
    (sequential? data) (str "[" (str/join "," (map ->json data)) "]")
    (number? data) (json-number data)
    (string? data) (str "\"" (-> data (str/replace "\\" "\\\\") (str/replace "\"" "\\\"") (str/replace "\n" "\\n")) "\"")
    (keyword? data) (->json (name data))
    (boolean? data) (str data)
    (nil? data) "null"
    :else (->json (str data))))
