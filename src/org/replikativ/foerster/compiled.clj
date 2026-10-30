(ns org.replikativ.foerster.compiled
  "A model whose structure is static given its data, compiled to plain
  functions of its latent values, and inference on them without worlds,
  savepoints or continuations.

  `compile-model` reads the model's instance graph (`analysis/instance`):
  every site with a concrete address, a distribution written over constants
  and the latent sites before it, and no site whose existence depends on a
  value. It returns its latent addresses, a log density of a double array
  of their values (the sum of every site's log density, as a run of the
  model would score it) and a draw from the prior. `rmh` runs `:rmh`'s
  moves (one latent drawn uniformly, a Gaussian step of `:step-size`) on
  that density: each move evaluates the density once, as Gen's dynamic MH
  re-runs its program once, but as compiled code.

  A model that does not qualify is refused with the reasons, so a caller
  can fall back to the general path."
  (:require [clojure.walk :as walk]
            [org.replikativ.foerster.analysis :as analysis]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.effects.savepoint :as sp]))

(defn- symbolic-address? [a]
  (some symbol? (tree-seq coll? seq a)))

(defn- dist-expr
  "The expression of site `s`'s distribution: its constructor applied to its
  arguments' expressions, or the expression it was given."
  [s]
  (let [{:keys [ctor args]} (:dist s)]
    (if ctor
      (when (every? #(contains? % :expr) args)
        (when (every? some? (map :expr args))
          (cons (symbol "org.replikativ.foerster.dist" (name ctor)) (map :expr args))))
      (:expr (first args)))))

(defn refusals
  "Why the instance graph `g` cannot be compiled: a vector of reasons, empty
  when it can."
  [{:keys [sites unknown result]}]
  (vec
   (concat
    (map (fn [u] {:reason :unknown :what (:reason u)}) unknown)
    (when (nil? result) [{:reason :unknown-result}])
    (for [[n s] (sort-by key sites)
          reason [(when (symbolic-address? (:address s)) :symbolic-address)
                  (when (seq (:control s)) :value-dependent-structure)
                  (when (seq (:path s)) :unknown-test)
                  (when (and (not= :factor (:kind s)) (nil? (dist-expr s))) :unknown-distribution)
                  (when (and (= :observed (:kind s)) (not (contains? (:observed s) :value))) :unknown-observation)
                  (when (and (= :latent (:kind s))
                             (not (#{:real :interval :positive :unit-interval}
                                   (get analysis/support (get-in s [:dist :ctor])))))
                    :not-continuous)]
          :when reason]
      {:reason reason :site n :address (:address s)}))))

(defn- code
  "`e` as code reading latent site n from `x` at its index in `index`."
  [e index x]
  (walk/postwalk (fn [f] (if (and (map? f) (contains? f :site) (= 1 (count f)))
                           (list 'clojure.core/aget x (index (:site f)))
                           f))
                 e))

(defn compile-model
  "The model applied to `args` compiled: {:addresses [a …] :log-density
  (fn [^doubles x]) :draw (fn [] ^doubles x)}, latent values in the order of
  :addresses. Throws ::not-static with the `refusals` when it cannot be."
  [model & args]
  (let [g (apply analysis/instance model args)
        why (refusals g)]
    (when (seq why)
      (throw (ex-info "The model's structure is not static given its data"
                      {:type ::not-static :refusals why})))
    (let [sites (sort-by key (:sites g))
          latents (vec (keep (fn [[n s]] (when (= :latent (:kind s)) n)) sites))
          index (zipmap latents (range))
          x (gensym "x")
          term (fn [[n s]]
                 (case (:kind s)
                   :factor (code (:expr (first (get-in s [:dist :args]))) index x)
                   :latent `(dist/logpdf ~(code (dist-expr s) index x) (aget ~x ~(index n)))
                   :observed `(dist/logpdf ~(code (dist-expr s) index x) '~(:value (:observed s)))))
          ns (the-ns (:ns (::analysis/source (meta model))))
          {:keys [params]} (::analysis/source (meta model))
          inputs (gensym "inputs")
          bind (vec (mapcat (fn [p i] [p `(nth ~inputs ~i)]) params (range)))
          density `(fn [~inputs]
                     (let ~bind
                       (fn [~(with-meta x {:tag 'doubles})]
                         (+ 0.0 ~@(map term sites)))))
          draw `(fn [~inputs]
                  (let ~bind
                    (fn []
                      (let [~x (double-array ~(count latents))]
                        ~@(for [n latents]
                            `(aset ~x ~(index n) (double (dist/draw ~(code (dist-expr ((:sites g) n)) index x)))))
                        ~x))))
          result (let [{:keys [value expr]} (:result g)]
                   (if expr
                     `(fn [~inputs] (let ~bind (fn [~(with-meta x {:tag 'doubles})] ~(code expr index x))))
                     `(fn [~'_] (fn [~'_] '~value))))
          compile (fn [form] ((binding [*ns* ns] (eval form)) (vec args)))]
      {:addresses (mapv #(:address ((:sites g) %)) latents)
       :log-density (compile density)
       :draw (compile draw)
       :result (compile result)})))

(defn rmh
  "Random-walk MH on a compiled model (`compile-model`), as `foerster.core`'s
  `:rmh`: `:chains` chains (default 4) of `:iterations` moves, each moving
  one latent drawn uniformly by a Gaussian step of `:step-size` (default
  0.1); every state after the first `:burn` moves is kept. Each chain draws
  from a stream of its own, seeded from `:seed` (else from the process
  generator, in order), and the chains run in parallel. Resolves, as a CPS
  operation, an EmpiricalMeasure of the model's results with `:chain-lengths`;
  a sample's trace holds its latent values only."
  [{:keys [addresses log-density draw result]}
   {:keys [iterations burn step-size chains seed] :or {burn 0 chains 4 step-size 0.1}}]
  (fn [resolve reject]
    (try
      (let [k (count addresses)
            seeds (if seed
                    (mapv #(sp/derive-seed seed ::chain %) (range chains))
                    (vec (repeatedly chains random/fresh-seed)))
            chain (fn []
                    (let [^doubles x (draw)]
                      (loop [i 0 lp (double (log-density x)) out (transient [])]
                        (if (= i iterations)
                          (persistent! out)
                          (let [j (int (* k (m/uniform01)))
                                old (aget x j)
                                _ (aset x j (+ old (* step-size (dist/draw (dist/normal 0.0 1.0)))))
                                lp' (double (log-density x))
                                accept? (and (== lp' lp') (< (Math/log (m/uniform01)) (- lp' lp)))
                                lp (if accept? lp' (do (aset x j old) lp))
                                out (if (>= i burn)
                                      (conj! out [(m/sample-particle (result x)
                                                                     (zipmap addresses (map (fn [v] {:value v}) x)))
                                                  0.0])
                                      out)]
                            (recur (inc i) lp out))))))
            chains (mapv deref (mapv (fn [s] (future (random/with-stream* s ::rmh chain))) seeds))]
        (resolve (assoc (m/markov-measure (into [] cat chains))
                        :chain-lengths (mapv count chains))))
      (catch Throwable t (reject t)))))
