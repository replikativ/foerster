(ns org.replikativ.foerster.amortized.compiler
  "Source-derived variable schemas and attention masks for GSDM tasks.
  Uses Foerster's existing program analysis. Factor-scope cliques are inference
  connectivity, not discovered causal edges. Fixed instances only; general
  simulator-boundary tasks remain available for opaque/dynamic internals."
  (:refer-clojure :exclude [compile await])
  (:require [clojure.set :as set]
            [clojure.repl :as repl]
            [clojure.walk :as walk]
            [org.replikativ.foerster.analysis :as analysis]
            [org.replikativ.foerster.amortized :as amortized]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as trace]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defmacro model
  "Write ordinary simulator code; this wrapper records source for analysis.
  Like analysis/model, the body is already inside a spin. No mask declaration."
  [args & body]
  `(analysis/model ~args ~@body))

(defn- canonical [x]
  (walk/postwalk #(if (and % (.isArray (class %))) (mapv canonical %) %) x))

(defn- source-digest [source args]
  (format "%064x" (java.math.BigInteger. 1
                                         (.digest (java.security.MessageDigest/getInstance "SHA-256")
                                                  (.getBytes (pr-str (canonical [source args])) "UTF-8")))))

(defn- source-from-var [v]
  (let [{:keys [ns name line]} (meta v)
        text (repl/source-fn (symbol (str (ns-name ns)) (str name)))]
    (when text
      (let [form (binding [*ns* ns] (read-string text))
            ;; source-fn starts at the defn; restore original source lines so
            ;; runtime trace locations can be checked against the analysis.
            form (walk/postwalk (fn [x]
                                  (if (:line (meta x))
                                    (vary-meta x update :line + (dec line)) x)) form)
            parts (drop-while #(or (string? %) (map? %)) (drop 2 form))]
        (when (and (#{"defn" "defn-"} (some-> form first clojure.core/name)) (vector? (first parts)))
          {:ns (ns-name ns) :params (first parts) :body (rest parts)})))))

(defn- known-argument [site i]
  (let [a (get-in site [:dist :args i])]
    (if (contains? a :value) (:value a)
        (when (:known? a) (let [form (:form a)] (when (coll? form) form))))))

(defn- infer-variable [name site]
  (let [ctor (get-in site [:dist :ctor]) support (analysis/support ctor)
        a (known-argument site 0) b (known-argument site 1)]
    (merge {:name name :role :variable}
           (case support
             :real {:type :real}
             :positive {:type :positive}
             :unit-interval {:type :interval :bounds [0.0 1.0]}
             :interval (when (and (number? a) (number? b)) {:type :interval :bounds [a b]})
             :real-vector (when (vector? a) {:type :real :shape [(count a)]})
             :simplex (when (vector? a) {:type :simplex :shape [(count a)]})
             :finite (when-let [values (case ctor
                                         flip [false true]
                                         bernoulli [0 1]
                                         binomial (when (and (integer? a) (<= 0 a 1024)) (vec (range (inc a))))
                                         uniform-discrete (when (and (integer? a) (integer? b) (< 0 (- b a) 1025)) (vec (range a b)))
                                         discrete (when (vector? a) (vec (range (count a))))
                                         categorical (cond (map? a) (vec (keys a))
                                                           (vector? a) (vec (distinct (map first a))))
                                         nil)]
                       {:type :finite :values values})
             nil))))

(defn- choose-entries [t]
  (vec (filter #(= trace/choose-site (:site %)) (trace/entries t))))

(defn- project [sites names t]
  (let [entries (choose-entries t)]
    (when-not (= (count sites) (count entries))
      (throw (ex-info "Simulator trace structure changed; use a boundary task or fixed/padded instance"
                      {:expected (count sites) :actual (count entries)})))
    (into {} (map (fn [[n site] entry]
                    (when (or (and (not (:structural? site)) (not= (:address site) (:address entry)))
                              (and (:loc site) (get-in entry [:note :loc])
                                   (not= (:loc site) (get-in entry [:note :loc]))))
                      (throw (ex-info "Runtime sites do not match the compiled simulator instance" {:site n})))
                    [(names n) (:value entry)]) sites entries))))

(defn- reconstruction [factory sites names]
  (fn [draw evidence {:keys [seed]}]
    (spin
     (let [values (merge draw evidence) cursor (atom 0)
           base (trace/policy)
           policy (fn [sp old]
                    (if (= trace/choose-site (:savepoint/site sp))
                      (let [i (dec (swap! cursor inc)) [n site] (nth sites i)
                            name (names n)]
                        (when-not (and (contains? values name) (= (:address site) (:savepoint/address sp)))
                          (throw (ex-info "Causal reconstruction needs every named stochastic site" {:site name})))
                        ;; Every supplied value is a factual assignment. Override
                        ;; observe's fixed-data mode without declaring a do-action.
                        ((trace/policy {:constraints {(:address site) (get values name)}})
                         (assoc-in sp [:savepoint/payload :observed?] false) old))
                      (base sp old)))
           t (await (gfi/run-policy (factory) policy {:seed seed}))
           result (try
                    (let [factual (project sites names t)]
                      (when-not (every? (fn [[name value]] (= value (get values name))) factual)
                        (throw (ex-info "Simulator changed a supplied factual assignment; use a free-coordinate causal adapter" {}))))
                    (cf/abduct-trace t 0.0)
                    (catch Throwable e (await (gfi/close! t)) (throw e)))]
       (await (gfi/close! t))
       result))))

(defn compile
  "Compile a source-backed defn Var, analysis/model or compiler/model into a task.
  Derives variable names/types and moralized factor-scope attention edges.
  Default :conditionings :any trains arbitrary observed subsets without 2^n
  enumeration. :args fixes loop sizes and model inputs. :schema-overrides
  handles unknown types/features. :causal? explicitly declares autonomous
  mechanisms and enables noise reconstruction; this initially needs named sites.
  Unknown control flow and unexpanded loops are refused, never guessed."
  ([model] (compile model {}))
  ([model {:keys [id args schema-overrides causal? targets known-inputs] :or {args []} :as options}]
   (let [reference model
         model (if (var? model) @model model)
         source (or (::analysis/source (meta model)) (when (var? reference) (source-from-var reference)))
         model (if source (with-meta model (assoc (meta model) ::analysis/source source)) model)]
     (when-not source
       (throw (ex-info "Pass a source-backed single-arity defn Var, or use compiler/model in the REPL, for automatic masks" {})))
     (let [graph (apply analysis/instance model args)
           all (:sites graph)
           sites (vec (sort-by key (filter (fn [[_ site]] (#{:latent :observed} (:kind site))) all)))
           _ (when (or (empty? sites) (seq (:unknown graph))
                       (some (comp seq :control second) sites)
                       (some #(and (seq (:sites %)) (not (:unrolled %))) (vals (:loops graph)))
                       (some #(= :factor (:kind %)) (vals all)))
               (throw (ex-info "Automatic masks currently require a fixed, factor-free source instance"
                               {:unknown (:unknown graph) :loops (:loops graph)})))
           aliases (when (map? (get-in graph [:result :expr]))
                     (into {} (for [[k expr] (get-in graph [:result :expr])
                                    :when (and (keyword? k) (map? expr) (contains? expr :site))]
                                [(:site expr) k])))
           names (into {} (for [[n site] sites]
                            [n (if (:structural? site) (or (get aliases n) (keyword (str "site-" n))) (:address site))]))
           _ (when-not (and (set/subset? (set targets) (set (vals names)))
                            (set/subset? (set known-inputs) (set (vals names)))
                            (empty? (set/intersection (set targets) (set known-inputs))))
               (throw (ex-info "Targets and known inputs must name disjoint compiled variables" {})))
           schema (mapv (fn [[n site]]
                          (let [role (cond (contains? (set known-inputs) (names n)) :input
                                           (nil? targets) :variable
                                           (contains? (set targets) (names n)) :target
                                           :else :observation)
                                v (merge (infer-variable (names n) site) {:role role} (get schema-overrides (names n)))]
                            (when-not (= (names n) (:name v))
                              (throw (ex-info "Schema overrides cannot rename compiled sites" {:name (names n)})))
                            (when-not (:type v)
                              (throw (ex-info "Cannot infer this RV's fixed representation; supply a schema override"
                                              {:name (names n) :distribution (:dist site)}))) v)) sites)
           conservative? (boolean (some (comp seq :escapes second) sites))
           edges (if conservative?
                   (vec (for [[n _] sites [m _] sites :when (< n m)] [(names n) (names m)]))
                   (vec (sort-by pr-str
                                 (set (mapcat (fn [[n site]]
                                                (let [scope (sort (filter names (conj (set/union (:reads site) (:control site)) n)))]
                                                  (for [a scope b scope :when (< a b)] [(names a) (names b)]))) sites)))))
           factory (fn [] (apply model args))
           digest (source-digest source args)
           _ (when (and causal? (some (comp :structural? second) sites))
               (throw (ex-info "Name causal sample sites with :id for stable noise replay" {})))
           task (amortized/task factory
                                (merge (dissoc options :args :schema-overrides :causal?)
                                       {:id (or id (keyword "amortized.compiled" (subs digest 0 16)))
                                        :schema schema :attention-edges edges
                                        :conditionings (or (:conditionings options) :any)
                                        :simulate-observed? true
                                        :semantics {:attention (if conservative? :dense-conservative :factor-scope-cliques) :source-sha256 digest
                                                    :causal? (boolean causal?) :structure :fixed-instance}
                                        :trace-project (partial project sites names)}))]
       (cond-> (assoc task :compilation {:sites (count sites) :edges (count edges) :names names
                                         :conservative? conservative?})
         causal? (assoc :model factory :reconstruct (reconstruction factory sites names)))))))
