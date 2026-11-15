(ns org.replikativ.foerster.runtime.async
  "Where foerster's code runs on engine v2.

  Inference drivers are host code: partial-cps `async` blocks returning CPS
  operations `(fn [resolve reject])`, which await with `pcps/await` and may
  call into any world (sessions, particle worlds). Models are v2 tasks. The
  savepoint and trace layers deliver as host code (`savepoint/off-world`).

  `op-of` makes a v2 handle a CPS operation, for a driver awaiting a task.
  `as-task` makes a driver's operation a task, for the public entry points:
  started off-world, so the driver never runs inside the caller's world.
  `await-op`, in a task body, awaits either."
  (:require [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.spindel.v2.host :as hs]
            [org.replikativ.spindel.v2.savepoint :as sp])
  #?(:cljs (:require-macros [org.replikativ.foerster.runtime.async]
                            [org.replikativ.spindel.v2.core :refer [task]])))

(defn op-of
  "`x` as a CPS operation: a v2 handle settles it once (its first settled
  value or error), delivered as host code (`savepoint/off-world`: never
  inside the handle's world, so the driver continuing may call into any
  world); anything else is taken as an operation already."
  [x]
  (if (v2/handle? x)
    (fn [resolve reject]
      (let [done (atom false)
            stop (volatile! nil)
            deliver! (sp/off-world (fn [[v err?]] (if err? (reject v) (resolve v))))
            cb (fn [v err?]
                 (when (compare-and-set! done false true)
                   (when-let [s @stop] (s))
                   (deliver! [v err?])))]
        (vreset! stop (v2/observe! x cb))
        (when @done (@stop))))
    x))

(defn now
  "`(apply f args)` as a CPS operation (a synchronous savepoint-layer call,
  e.g. `sp/fork`, where an operation is expected); a throw rejects."
  [f & args]
  (fn [resolve reject]
    (let [[ok v] (try [true (apply f args)]
                      (catch #?(:clj Throwable :cljs :default) e [false e]))]
      (if ok (resolve v) (reject v)))))

(defn external-of
  "The `v2/external` operation completing with CPS operation `op`, started
  on the host executor: never inside the world whose body awaits it. That
  world is bound (`v2/*world*`) while `op` starts, so a driver may take it
  as its caller's world (`with-world`)."
  [op]
  (fn [complete!]
    (let [w (v2/current-world)]
      (hs/execute! (hs/executor)
                   (fn [] (binding [v2/*world* w]
                            (try (op complete! (fn [e] (complete! e true)))
                                 (catch #?(:clj Throwable :cljs :default) e (complete! e true)))))))
    nil))

(defn with-world
  "`(f)` with `v2/*world*` bound to `w`: a driver calls a user function that
  may create tasks (a backend, a validator) in its caller's world `w`,
  taken where the driver started, also after it awaited (as host code no
  world is bound)."
  [w f]
  (binding [v2/*world* w] (f)))

#?(:clj
   (defmacro await-op
     "In a task body: the value of `x`, a v2 handle or a CPS operation."
     [x]
     `(let [x# ~x]
        (if (v2/handle? x#)
          (v2/await x#)
          (v2/external (external-of x#))))))

(defn as-task
  "A task (in the current world: a body's, or the bound `v2/*world*`)
  resolving the CPS operation `op`, which runs as host code."
  [op]
  ;; `await-op` spelled out: ClojureScript cannot expand this namespace's own
  ;; macro here
  (task (if (v2/handle? op) (v2/await op) (v2/external (external-of op)))))
