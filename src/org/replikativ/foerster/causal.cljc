(ns org.replikativ.foerster.causal
  "Graphical identification in declared acyclic directed mixed graphs (ADMGs).
  Directed edges are causal assumptions; bidirected edges represent latent
  common causes. This is not a causal interpretation inferred from source.

  `identify` implements Shpitser and Pearl's ID/IDC algorithms (2006).
  It returns an observational functional or a nonidentification witness.
  Identification assumes the graph, its latent-variable interpretation and
  the positivity required by the functional. It does not fit mechanisms.
  Expressions are data and can be evaluated on finite probability tables."
  (:refer-clojure :exclude [ancestors descendants])
  (:require [clojure.set :as set]))

(defn- ordered [xs] (vec (sort-by pr-str xs)))

(defn topological-order [{:keys [nodes directed]}]
  (loop [remaining nodes out []]
    (if (empty? remaining)
      out
      (let [roots (ordered (remove (fn [v] (some #(and (= v (second %))
                                                       (remaining (first %))) directed)) remaining))]
        (when (empty? roots)
          (throw (ex-info "Directed causal edges must be acyclic" {:type ::cycle})))
        (recur (apply disj remaining roots) (into out roots))))))

(defn graph
  "Declare {:nodes #{…} :directed [[cause effect] …] :bidirected [[a b] …]}.
  Nodes can be inferred from edges, but isolated nodes must be supplied."
  [{:keys [nodes directed bidirected]}]
  (let [edges (concat directed bidirected)
        nodes (or nodes (set (mapcat identity edges)))
        g {:nodes (set nodes) :directed (set (map vec directed))
           :bidirected (set (map set bidirected))}]
    (when-not (every? #(and (some? %) (not (false? %))) nodes)
      (throw (ex-info "Causal node identifiers cannot be nil or false" {:type ::invalid-graph})))
    (when-not (every? #(and (= 2 (count %)) (not= (first %) (second %))
                            (every? (:nodes g) %)) edges)
      (throw (ex-info "Edges need two distinct declared nodes" {:type ::invalid-graph})))
    (topological-order g)
    g))

(defn subgraph [g nodes]
  (let [nodes (set nodes)]
    (assoc g :nodes nodes
           :directed (set (filter #(every? nodes %) (:directed g)))
           :bidirected (set (filter #(every? nodes %) (:bidirected g))))))

(defn ancestors
  "Directed ancestors, including `nodes` themselves."
  [g nodes]
  (loop [seen (set nodes)]
    (let [more (into seen (keep (fn [[a b]] (when (seen b) a))) (:directed g))]
      (if (= seen more) seen (recur more)))))

(defn descendants [g nodes]
  (ancestors (update g :directed #(set (map (comp vec reverse) %))) nodes))

(defn districts
  "Maximal bidirected-connected components, including isolated nodes."
  [g]
  (loop [remaining (:nodes g) out []]
    (if (empty? remaining)
      out
      (let [component (loop [seen #{(first (ordered remaining))}]
                        (let [more (reduce (fn [s edge] (if (seq (set/intersection s edge))
                                                          (set/union s edge) s)) seen (:bidirected g))]
                          (if (= seen more) seen (recur more))))]
        (recur (set/difference remaining component) (conj out component))))))

(defn m-separated?
  "m-separation by latent expansion and ancestral moralization."
  [g left right conditioned]
  (let [latent (map-indexed (fn [i edge] [[::latent i (gensym "u")] edge]) (:bidirected g))
        expanded {:nodes (into (:nodes g) (map first latent))
                  :directed (into (:directed g) (mapcat (fn [[u edge]] (map #(vector u %) edge)) latent))}
        relevant (ancestors expanded (set/union (set left) (set right) (set conditioned)))
        edges (:directed (subgraph expanded relevant))
        parents (group-by second edges)
        moral (concat edges (mapcat (fn [ps] (for [[a _] ps [b _] ps :when (not= a b)] [a b]))
                                    (vals parents)))
        adjacency (reduce (fn [a [u v]] (-> a (update u (fnil conj #{}) v)
                                            (update v (fnil conj #{}) u))) {} moral)
        blocked (set conditioned) targets (set right)]
    (loop [todo (set/difference (set left) blocked) seen blocked]
      (cond
        (seq (set/intersection todo targets)) false
        (empty? todo) true
        :else (let [v (first todo)]
                (recur (set/union (disj todo v) (set/difference (get adjacency v #{}) (conj seen v)))
                       (conj seen v)))))))

(defn- probability [vars given]
  (if (empty? vars) 1 {:op :probability :vars (vec vars) :given (vec given)}))

(defn- product [terms]
  (let [terms (vec (remove #{1} terms))]
    (case (count terms) 0 1 1 (first terms) {:op :product :terms terms})))

(defn- marginal [vars expr]
  (let [vars (set vars)]
    (cond
      (empty? vars) expr
      (number? expr) expr
      (and (= :probability (:op expr)) (empty? (:given expr)))
      (probability (remove vars (:vars expr)) [])
      (= :sum (:op expr)) (marginal (set/union vars (set (:over expr))) (:expr expr))
      :else {:op :sum :over (ordered vars) :expr expr})))

(defn- ratio [a b] (if (= b 1) a {:op :ratio :numerator a :denominator b}))

(defn- chain-factor [p order v]
  (let [before (vec (take-while #(not= % v) order))
        all (set order)]
    (if (and (= :probability (:op p)) (empty? (:given p)) (= all (set (:vars p))))
      (probability [v] before)
      (ratio (marginal (set/difference all (conj (set before) v)) p)
             (marginal (set/difference all (set before)) p)))))

(declare id*)

(defn- id* [g y x p]
  (let [v (:nodes g) an (ancestors g y)]
    (cond
      (empty? x) (marginal (set/difference v y) p)
      (not= an v) (id* (subgraph g an) y (set/intersection x an)
                       (marginal (set/difference v an) p))
      :else
      (let [without-x (subgraph g (set/difference v x))
            w (set/difference (:nodes without-x) (ancestors without-x y))
            components (districts without-x)]
        (cond
          (seq w) (id* g y (set/union x w) p)
          (> (count components) 1)
          (marginal (set/difference v (set/union y x))
                    (product (map #(id* g % (set/difference v %) p) components)))
          :else
          (let [s (first components) all-components (districts g)
                order (topological-order g)]
            (cond
              (= all-components [v])
              (throw (ex-info "Effect is not identified from the observational law"
                              {:type ::not-identified :witness {:outer g :inner (subgraph g s)}}))
              (some #{s} all-components)
              (marginal (set/difference s y) (product (map #(chain-factor p order %) (filter s order))))
              :else
              (let [larger (first (filter #(set/subset? s %) all-components))]
                (id* (subgraph g larger) y (set/intersection x larger)
                     (product (map #(chain-factor p order %) (filter larger order))))))))))))

(defn- remove-incoming [g x]
  (assoc g :directed (set (remove #(x (second %)) (:directed g)))
         :bidirected (set (remove #(seq (set/intersection x %)) (:bidirected g)))))

(defn- idc* [g y x z p]
  (if (empty? z)
    (id* g y x p)
    (if-let [move (first (filter (fn [node]
                                   (let [g' (update (remove-incoming g x) :directed
                                                    #(set (remove (fn [[a _]] (= a node)) %)))]
                                     (m-separated? g' y #{node} (set/union x (disj z node)))))
                                 (ordered z)))]
      (idc* g y (conj x move) (disj z move) p)
      (let [joint (id* g (set/union y z) x p)]
        (ratio joint (marginal y joint))))))

(defn identify
  "Identify P(outcomes | do(treatments), given) from P(V) in an ADMG.
  Returns :identified with :expression, or :not-identified with a witness.
  Scope is nonparametric observational identification in a semi-Markovian
  acyclic model; it is not an algebraic parameter-identifiability verdict."
  [g {:keys [outcomes treatments given]}]
  (let [g (graph (update g :bidirected #(map vec %)))
        y (set outcomes) x (set treatments) z (set given)]
    (when-not (and (seq y) (set/subset? (set/union y x z) (:nodes g))
                   (empty? (set/intersection y x)) (empty? (set/intersection y z))
                   (empty? (set/intersection x z)))
      (throw (ex-info "Query sets must be declared, disjoint, with nonempty outcomes"
                      {:type ::invalid-query})))
    (try
      {:status :identified :scope :observational-nonparametric
       :expression (idc* g y x z (probability (topological-order g) []))}
      (catch #?(:clj clojure.lang.ExceptionInfo :cljs ExceptionInfo) e
        (if (= ::not-identified (:type (ex-data e)))
          {:status :not-identified :scope :observational-nonparametric :witness (:witness (ex-data e))}
          (throw e))))))

(defn evaluate
  "Evaluate an identified expression on a finite probability oracle.
  `probability` takes maps of event values and conditioning values.
  `domains` gives finite supports for summed variables. Undefined ratios
  (including positivity failures) are refused. Continuous integrals need
  a different evaluator; symbolic identification itself is not discrete."
  [expression domains probability assignment]
  (letfn [(ev [e a]
            (if (number? e) e
                (case (:op e)
                  :probability
                  (do (when-not (every? #(contains? a %) (concat (:vars e) (:given e)))
                        (throw (ex-info "Expression has unassigned variables" {:type ::missing-assignment :expression e})))
                      (let [p (probability (select-keys a (:vars e)) (select-keys a (:given e)))]
                        (when-not (and (number? p) (<= 0 p 1))
                          (throw (ex-info "Probability oracle must return a number in [0,1]"
                                          {:type ::invalid-probability :value p :expression e})))
                        p))
                  :product (reduce * (map #(ev % a) (:terms e)))
                  :ratio (let [den (ev (:denominator e) a)]
                           (when-not (pos? den)
                             (throw (ex-info "Observational functional violates positivity"
                                             {:type ::positivity :expression e :assignment a})))
                           (/ (ev (:numerator e) a) den))
                  :sum (letfn [(sum-over [vars a]
                                 (if-let [[v & more] (seq vars)]
                                   (let [support (get domains v)]
                                     (when-not (seq support)
                                       (throw (ex-info "Summation needs a finite domain" {:type ::missing-domain :node v})))
                                     (reduce + (map #(sum-over more (assoc a v %)) support)))
                                   (ev (:expr e) a)))]
                         (sum-over (:over e) a)))))]
    (ev expression assignment)))
