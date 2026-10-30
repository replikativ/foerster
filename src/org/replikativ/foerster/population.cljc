(ns org.replikativ.foerster.population
  "Particle populations on θ vectors: the shared engine of the samplers that
  move a model that is one block (`foerster.block`) without worlds.

  A model that is one latent block site and nothing else is, for inference,
  its θ: its density is the block's, and the program around it only computes
  the result from θ. Such a model runs here on plain vectors, and a world is
  made only at the end, by one replay per distinct final θ (`write-back-all`),
  so every result is still computed by the program itself.

  `theta-smc` is an SMC sampler over a sequence of targets on θ, given by a
  path: tempering from the block's draw density to its target is one path
  (`foerster.tempering`); data arriving in batches or a hierarchy of model
  levels are others. A path decides the next position from the population,
  gives each particle's incremental log weight to it, and the log density the
  moves at that position leave invariant. Resampling, the moves and the
  evidence are the engine's.

  Randomness is keyed by what is drawn (`foerster.random`): the resampling
  at step k and the moves of chain i at step k read their own streams, under
  keys the caller names, so a sampler moved onto this engine keeps its
  draws."
  (:require [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.population.path :as path]
            [org.replikativ.foerster.execution :as execution]
            [org.replikativ.foerster.block-mh :as block-mh]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.trace :as trace]))

(defn relative-cess
  "The conditional ESS of the incremental weights exp(δ·ℓ) under the
  normalized log weights `lw`, as a fraction of N."
  [lw ls delta]
  (let [a (mapv (fn [w l] (+ w (* delta l))) lw ls)
        b (mapv (fn [w l] (+ w (* 2.0 delta l))) lw ls)
        r (- (* 2.0 (weights/log-sum-exp a)) (weights/log-sum-exp b))]
    (if (#?(:clj Double/isNaN :cljs js/isNaN) r) 0.0 (Math/exp r))))

(defn next-delta
  "The temperature increment whose conditional ESS is `target`: the rest of
  the way to 1 when that keeps it, else by bisection."
  [lw ls beta target]
  (let [rest (- 1.0 beta)]
    (if (>= (relative-cess lw ls rest) target)
      rest
      (loop [lo 0.0 hi rest i 0]
        (let [mid (* 0.5 (+ lo hi))]
          (if (or (= i 60) (< (- hi lo) 1e-12))
            (max mid 1e-12)
            (if (>= (relative-cess lw ls mid) target)
              (recur mid hi (inc i))
              (recur lo mid (inc i)))))))))

(defn executor-of [trace] (:executor (:trace/world trace)))

(defn single-block
  "{:address :dist} when every trace in `ts` is one latent block site and
  nothing else (no other choice, observation or factor), all under one law:
  then a particle is its θ, and inference needs no world until the end."
  [ts]
  (let [shape (fn [t]
                (let [es (itrace/entries t)]
                  (when (and (= 1 (count es)) (itrace/latent? (first es))
                             (block-mh/block-site? t (:address (first es))))
                    {:address (:address (first es)) :dist (:dist (:note (first es)))})))
        s (shape (first ts))]
    (when (and s (every? #(= s (shape %)) (rest ts)))
      s)))

(defn theta-scales
  "A random walk over θ vectors sized to the population: each coordinate's
  standard deviation over `qs` times `scale`/√dimension."
  [qs scale]
  (let [n (count qs) dim (count (first qs))
        mean (apply mapv (fn [& xs] (/ (reduce + xs) n)) qs)
        sd (apply mapv (fn [mu & xs]
                         (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
                                       (max 1 (dec n)))))
                  mean qs)]
    (mapv #(* (/ scale (Math/sqrt dim)) (max % 1e-9)) sd)))

(defn write-back-all
  "Traces of `template` with the block site at `address` set to each of `qs`
  (one replay per distinct θ); resolves them in order, or rejects when the
  program fails on one of them."
  [template address qs]
  (fn [resolve reject]
    (let [distinct-qs (vec (distinct qs))
          policy (fn [q] (itrace/policy {:constraints {address q}}))]
      ((execution/parallel (executor-of template) (count distinct-qs)
                           (fn [i res rej]
                             ((trace/replay template address (policy (nth distinct-qs i)) {:anchor? itrace/anchor?})
                              res rej)))
       (fn [ts]
         (if-let [error (some :trace/error ts)]
           (reject (ex-info "The program failed on a final state" {:type ::model-failed} error))
           (let [by-q (zipmap distinct-qs ts)] (resolve (mapv by-q qs)))))
       reject))))

(defn theta-smc
  "An SMC sampler on θ vectors `qs` along `path`; returns {:qs :log-z
  :positions}.

  `path`:
    :start      the first position
    :done?      (fn [pos]) whether `pos` is the last target
    :advance    (fn [qs lw pos]) -> {:pos pos' :log-incr [w …]}: the next
                position and every particle's incremental log weight to it
    :density    (fn [pos]) -> the log density of θ the moves at `pos` leave
                invariant
    :exhausted  (fn [pos]) -> the exception thrown after `:max-steps`

  `opts`: `:n` particles, `:waste-free` P (resample n/P and keep every state
  of their P-step chains), `:scale` of the population random walk,
  `:max-steps`, `:seed`, `:stats` (an atom counting `:moves` and
  `:accepted`), `:keys` {:resample k :move k}, the stream keys of step
  k's resampling ([k-resample k]) and chain i's moves ([k-move k i]), and
  `:positions?` (true): false keeps only the last position in `:positions`."
  [qs path {:keys [n waste-free scale max-steps seed stats keys positions?] :or {positions? true}}]
  (let [dim (count (first qs))
        backend {:move
                 (fn [starts position k steps]
                   (fn [resolve reject]
                     (try
                       (let [scales (theta-scales starts scale)
                             f ((:density path) position)
                             chains (mapv
                                     (fn [i q0]
                                       (random/with-stream* seed [(:move keys) k i]
                                         #(loop [s 0 q q0 states []]
                                            (if (= s steps) states
                                                (let [{q' :q accepted :accepted} (block-mh/walk f q scales dim)]
                                                  (swap! stats (fn [st] (-> st (update :moves + dim)
                                                                            (update :accepted + accepted))))
                                                  (recur (inc s) q' (conj states q')))))))
                                     (range) starts)]
                         (resolve {:chains chains}))
                       (catch #?(:clj Throwable :cljs :default) e (reject e)))))}
        result (path/run-sync
                (path/run {:particles (vec qs) :mass (weights/uniform (count qs)) :position (:start path)}
                          path backend {:n n :waste-free waste-free :seed seed :keys keys
                                        :max-steps max-steps :positions? positions?}))]
    {:qs (:particles result) :log-z (get-in result [:mass :log-mass]) :positions (:positions result)}))
