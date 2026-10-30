(ns hooks.foerster
  (:require [clj-kondo.hooks-api :as api]))

(defn scoped-bindings
  "The fixture binds each name and uses it in resource setup/finalization.
  Preserve its body and lexical scope without inventing runtime fixture code."
  [{:keys [node]}]
  (let [[_ bindings & body] (:children node)
        names (:children bindings)]
    {:node
     (api/list-node
      (concat [(api/token-node 'fn)
               bindings
               ;; Fixture setup/finalization uses every bound resource.
               (api/vector-node names)]
              body))}))
