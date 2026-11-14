(ns org.replikativ.foerster.runtime.async
  "Awaiting inference operations in a v2 task body.

  Foerster's drivers are partial-cps `async` blocks and savepoint-layer
  operations: CPS operations `(fn [resolve reject])`. A task body awaits one
  through v2's `external` effect, and a v2 handle (task, computation,
  deferred) with `await`. `await-op` takes either; it is a macro so that the
  effect stays lexical in the body."
  (:require [org.replikativ.spindel.v2.core :as v2])
  #?(:cljs (:require-macros [org.replikativ.foerster.runtime.async])))

(defn external-of
  "The `v2/external` operation completing with CPS operation `op`."
  [op]
  (fn [complete!]
    (op complete! (fn [e] (complete! e true)))
    nil))

#?(:clj
   (defmacro await-op
     "In a task body: the value of `x`, a v2 handle or a CPS operation."
     [x]
     `(let [x# ~x]
        (if (v2/handle? x#)
          (v2/await x#)
          (v2/external (external-of x#))))))
