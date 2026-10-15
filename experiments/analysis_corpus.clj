(ns analysis-corpus
  "Run foerster.analysis over every spin form with a site in the given
  files; print a summary per model and totals.

    clojure -Sdeps '{:aliases {:corpus {:extra-paths [\"test\" \"experiments\"]}}}' \\
      -M:corpus -m analysis-corpus test/org/replikativ/foerster/*_test.clj"
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.spindel.spin.cps]))

(defn- site-call? [ns x]
  (and (seq? x) (symbol? (first x))
       (let [v (try (ns-resolve ns (first x)) (catch Exception _ nil))]
         (and (var? v) (= "org.replikativ.foerster.effects" (str (.-ns ^clojure.lang.Var v)))
              (#{"sample" "observe" "factor" "choose"} (str (.-sym ^clojure.lang.Var v)))))))

(defn- spin? [ns x]
  (and (seq? x) (symbol? (first x))
       (= #'org.replikativ.spindel.spin.cps/spin (try (ns-resolve ns (first x)) (catch Exception _ nil)))))

(defn- outermost-spins [ns form]
  (if (spin? ns form)
    [form]
    (when (coll? form) (mapcat #(outermost-spins ns %) (if (map? form) (concat (keys form) (vals form)) form)))))

(defn- forms [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (let [first-form (read {:eof ::eof :read-cond :allow :features #{:clj}} r)
          ns-sym (second first-form)]
      (require ns-sym)
      (binding [*ns* (the-ns ns-sym)]
        [(the-ns ns-sym)
         (doall (take-while #(not= ::eof %)
                            (repeatedly #(read {:eof ::eof :read-cond :allow :features #{:clj}} r))))]))))

(defn -main [& files]
  (let [totals (atom {})]
    (doseq [f files]
      (try
        (let [[ns fs] (forms f)]
          (doseq [form fs
                  s (outermost-spins ns form)
                  :when (some #(site-call? ns %) (tree-seq coll? seq s))]
            (let [g (try (a/analyze-form s {:ns ns}) (catch Throwable t {:error (.getMessage t)}))]
              (if (:error g)
                (do (swap! totals update :errors (fnil inc 0))
                    (println "ERROR" f (:line (meta s)) (:error g)))
                (let [fs (a/facts g)
                      kinds (cond-> (frequencies (map :fact fs))
                              (some :escapes fs) (assoc :escaping (count (filter :escapes fs))))]
                  (swap! totals #(merge-with + % {:models 1 :sites (count (:sites g))} (dissoc kinds :latent :markov-blanket)))
                  (println (str (.getName (io/file f)) ":" (:line (meta s)))
                           (count (:sites g)) "sites"
                           (pr-str (dissoc kinds :latent :markov-blanket))))))))
        (catch Throwable t (println "SKIP" f (.getMessage t)))))
    (println "TOTAL" (pr-str @totals))))
