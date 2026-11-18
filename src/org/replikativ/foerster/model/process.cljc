(ns org.replikativ.foerster.model.process
  "Memoization and random processes whose state lives in the particle's
  world, so it forks with the particle and is discarded with it — Anglican's
  `mem` and its Chinese restaurant process, on spindel worlds.

  `mem` turns a task-returning function into one that computes each
  argument list once per world: the first call samples, later calls (in the
  same particle, and in every particle forked after it) return that value.
  Nonparametric models are written with it, as a stick per index or a
  mean per cluster:

    (let [mean-of (process/mem (fn [k] (task (sample (dist/normal 0 10) :id [:mean k]))))]
      (task … (await (mean-of 3)) …))

  Name the sample sites inside a memoized function by its arguments, as
  above: a value is drawn once, at the first call.

  `crp-draw` seats a customer in a Chinese restaurant process: table k with
  probability ∝ its customers, a new table ∝ α. The tables live in the world
  under the process's name; the draw is an ordinary sample site over the
  existing tables and a new one, so every method — exact enumeration
  included — handles it."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.spindel.v2.core :as v2 :refer [task await]])
  #?(:cljs (:require-macros [org.replikativ.spindel.v2.core :as v2 :refer [task await]])))

(defn- world [] (v2/current-world))

(defn mem
  "`f` (a function returning a task or a value) memoized per world: each
  argument list is computed once in a particle, and particles forked later
  share what was computed before the fork. Returns a function of the same
  arguments returning a task."
  [f]
  (let [id #?(:clj (Object.) :cljs (js-obj))]
    (fn [& args]
      (task
       (let [key [::mem id (vec args)]
             ;; boxed, so a memoized nil is told from a missing value
             cached (v2/get-in-world (world) [:inference :mem key])]
         (if cached
           (first cached)
           (let [out (apply f args)
                 v (if (v2/handle? out) (await out) out)]
             (v2/update-in-world! (world) [:inference :mem key] (constantly [v]))
             v)))))))

(defn crp-tables
  "The customers at each table of the process `name` in the current world:
  a vector, table k at index k."
  [name]
  (or (v2/get-in-world (world) [:inference :crp name]) []))

(defn crp-draw
  "Seat a customer in the Chinese restaurant process `name` with
  concentration `alpha` > 0: a task resolving the table index, an existing
  table k with probability ∝ its customers, a new table (index = the number
  of tables) ∝ α. `:id` names the sample site."
  [name alpha & {:keys [id]}]
  (task
   (let [tables (crp-tables name)
         k (sample (dist/discrete (conj tables alpha)) :id id)]
     (v2/update-in-world! (world) [:inference :crp name]
                      (fn [ts] (let [ts (or ts [])] (if (= k (count ts)) (conj ts 1) (update ts k inc)))))
     k)))
