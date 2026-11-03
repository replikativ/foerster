(ns org.replikativ.foerster.layout-test
  "The source layout's layering, read from the ns forms: a namespace may
  require only namespaces of its own layer or below. A failure names the
  offending require; move the code, or the namespace, rather than the rank."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]))

(def ^:private prefix "org.replikativ.foerster.")

(def layers
  "Rank of each layer, by the namespace's first segment below foerster (a
  top-level namespace by its own name). Equal ranks may require each other."
  {"random" 0
   "dist" 1 "weights" 1
   "measure" 2 "effects" 2 "block" 2 "diagnostics" 3
   "model" 4
   "runtime" 5
   "analysis" 6
   "moves" 7
   "population" 8
   "inference" 9
   "causal" 10
   "core" 11 "program" 11
   "amortized" 12 "sandbox" 12})

(defn- layer [ns-name] (first (str/split (subs ns-name (count prefix)) #"\.")))

(defn- source-namespaces []
  (for [f (file-seq (io/file "src/org/replikativ/foerster"))
        :when (re-find #"\.clj[cs]?$" (.getName ^java.io.File f))
        :let [form (with-open [r (java.io.PushbackReader. (io/reader f))]
                     (binding [*read-eval* false]
                       (read {:read-cond :allow :features #{:clj}} r)))
              ;; an :as-alias libspec loads nothing (it names keywords): not
              ;; a dependency, so it is not descended into
              requires (->> (tree-seq #(and (coll? %) (not (and (vector? %) (some #{:as-alias} %))))
                                      seq form)
                            (filter symbol?)
                            (map str)
                            (filter #(str/starts-with? % prefix))
                            set)]]
    {:ns (str (second form)) :requires (disj requires (str (second form)))}))

(deftest every-layer-is-ranked
  (doseq [{n :ns} (source-namespaces)]
    (is (contains? layers (layer n)) (str n " has no layer in layout-test/layers"))))

(deftest namespaces-require-only-their-layer-and-below
  (doseq [{n :ns requires :requires} (source-namespaces)
          r requires
          :let [rn (layers (layer n)) rr (layers (layer r))]
          :when (and rn rr)]
    (is (<= rr rn) (str n " (" (layer n) ") requires " r " (" (layer r) "), a higher layer"))))
