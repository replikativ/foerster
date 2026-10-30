(ns org.replikativ.foerster.analysis
  "The structure of a model, read from its source: which sites it has, what
  each one reads, and the facts an inference plan can be derived from.

  `analyze` walks the macro-expanded body of a model (a `spin` form, or the
  forms inside one) and returns its site graph:

    {:sites   {n {:kind :latent | :observed | :factor
                  :address [:y i]        ; the :id form, symbols are loop variables;
                                         ; [:loc line col] for a site without :id
                  :dist {:ctor normal :args [{:form mu :reads #{0} :exact 0} …]}
                  :reads #{0}            ; sites its distribution's arguments read
                  :control #{}           ; sites that decide whether it runs
                  :loops [2]}}           ; enclosing loops, innermost last
     :loops   {2 {:bindings [i] :sites #{1} :carried #{} :kind :plate | :sequential}}
                 ; :carried: its sites a later iteration reads
     :unknown [{:reason :closure :form …}]}

  A site's value reads the sites its arguments were computed from: the
  analysis follows locals through let, loop, if and calls, over-approximating
  a call's result as reading everything its arguments read. A loop whose
  recurring values read none of its own sites runs independent iterations (a
  plate); otherwise it is sequential. What the walk cannot see — a site in a
  closure, which the spin macro cannot see either — is listed under
  `:unknown`, never guessed.

  Every value also carries its expression of the model's inputs, constants
  and sites (a site's argument's `:expr`, sites written {:site n}) as far as
  the walk can write it: through `pure-fns` and `Math` calls, `if`, and
  collections. That is the argument's link function; `affine` reads a + b·x
  off it.

  `facts` derives from the graph: the support of each site, finite latent
  sites, plates, conjugate pairs (`foerster.conjugate`'s families, and a
  Normal mean affine in a Normal prior, `:normal-affine` with each child's
  link and the sites it is conditional on) and Markov blankets. A latent
  site whose value, or a value computed from it, goes anywhere but a
  distribution's argument or the model's result — an impure call, a
  closure, the next iteration — escapes and is not collapsed. `explain`
  prints both."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            ;; `model` expands to a spin
            [org.replikativ.spindel.spin.cps]))

;; -----------------------------------------------------------------------------
;; What the walk recognizes
;; -----------------------------------------------------------------------------

(def ^:private site-fns
  '{org.replikativ.foerster.effects/sample :latent
    org.replikativ.foerster.core/sample :latent
    org.replikativ.foerster.effects/observe :observed
    org.replikativ.foerster.core/observe :observed
    org.replikativ.foerster.effects/choose :choose
    org.replikativ.foerster.effects/factor :factor
    org.replikativ.foerster.core/factor :factor})

(def ^:private spin-macros
  '#{org.replikativ.spindel.spin.cps/spin})

(def ^:private dist-ns "org.replikativ.foerster.dist")

(def support
  "The support of a distribution, by its constructor in `foerster.dist`."
  '{normal :real, cauchy :real, student-t :real, mvn :real-vector
    uniform :interval
    exponential :positive, gamma :positive, half-normal :positive,
    half-cauchy :positive, log-normal :positive, chi-squared :positive
    beta :unit-interval, dirichlet :simplex
    poisson :count, negative-binomial :count
    bernoulli :finite, flip :finite, binomial :finite, discrete :finite,
    categorical :finite, uniform-discrete :finite})

(def continuous #{:real :real-vector :interval :positive :unit-interval :simplex})

(def conjugate-families
  "Prior constructor → likelihood constructor → [the likelihood argument the
  parameter must be, exactly; `foerster.conjugate` family]."
  '{beta {bernoulli [0 :beta-bernoulli] flip [0 :beta-bernoulli] binomial [1 :beta-binomial]}
    gamma {poisson [0 :gamma-poisson]}
    normal {normal [0 :normal-mean]}
    dirichlet {discrete [0 :dirichlet-discrete]}})

(def implemented-families
  "Families `foerster.conjugate` implements."
  #{:normal-mean :beta-bernoulli :gamma-poisson :dirichlet-discrete})

;; -----------------------------------------------------------------------------
;; The walk
;; -----------------------------------------------------------------------------

;; A value is {:reads #{site …} :exact site-or-nil :expr e}: the sites it was
;; computed from, the site whose value it is, unchanged, if it is one, and the
;; expression it is of the model's inputs, constants and sites — its link
;; function — when the walk can write it (else nil). In an expression a
;; latent site n is {:site n}; operators are qualified symbols.
(def ^:private const {:reads #{} :exact nil :expr nil})

(defn- reads [& vs] {:reads (reduce set/union #{} (map :reads vs)) :exact nil :expr nil})

(def pure-fns
  "Functions whose result is their expression of their arguments: a value
  passed to one does not go anywhere else (`facts`' escape check), and its
  result has an expression (`affine`)."
  (into '#{clojure.core/+ clojure.core/- clojure.core/* clojure.core// clojure.core/inc
           clojure.core/dec clojure.core/max clojure.core/min clojure.core/abs clojure.core/double
           clojure.core/long clojure.core/int clojure.core/float clojure.core/not clojure.core/=
           clojure.core/not= clojure.core/< clojure.core/> clojure.core/<= clojure.core/>=
           clojure.core/zero? clojure.core/pos? clojure.core/neg? clojure.core/nth clojure.core/get
           clojure.core/first clojure.core/second clojure.core/count clojure.core/vector
           clojure.core/conj clojure.core/assoc clojure.core/identity clojure.core/hash-map
           clojure.core/sqrt clojure.core/exp clojure.core/log}
        (map #(symbol "org.replikativ.foerster.dist" (name %)))
        (keys support)))

(defn- static-math? [f]
  (and (symbol? f) (#{"Math" "java.lang.Math"} (namespace f))))

(defn- scalar? [v]
  (or (number? v) (keyword? v) (boolean? v) (string? v) (nil? v) (char? v)))

(defn- known
  "The value of a constant `v`, written `expr` in expressions (`v` itself if
  it is a scalar)."
  [v expr]
  {:reads #{} :exact nil :expr (if (scalar? v) v expr) :known? true :value v})

(defn- invoke
  "The result of the pure function `q` (or static `Math` method `f`) on
  `args`, or ::fail."
  [q f args]
  (try
    (if q
      (apply @(find-var q) args)
      (clojure.lang.Reflector/invokeStaticMethod "java.lang.Math" (name f) (object-array args)))
    (catch Exception _ ::fail)))

(defn- resolve-sym
  "The fully qualified symbol `sym` names in `ns`, or nil for a local, a
  special form, a class or an unresolvable name."
  [ns locals sym]
  (when (and (symbol? sym) (not (contains? locals sym)))
    (let [v (try (ns-resolve ns sym) (catch Exception _ nil))]
      (when (var? v) (symbol (str (.-ns ^clojure.lang.Var v)) (str (.-sym ^clojure.lang.Var v)))))))

(defn- macro? [ns locals sym]
  (when (and (symbol? sym) (not (contains? locals sym)))
    (let [v (try (ns-resolve ns sym) (catch Exception _ nil))]
      (and (var? v) (:macro (meta v))))))

(defn- binding-syms
  "The symbols a let*/loop* binding vector binds (already destructured by the
  macros that expand to them)."
  [bvec]
  (take-nth 2 bvec))

(defn- options [opts]
  (when (even? (count opts)) (apply hash-map opts)))

(declare walk)

(defn- escape!
  "Note that the value `v` — a latent site's value, or one computed from it
  by `pure-fns` — goes somewhere other than a distribution's argument or the
  model's result: every site it reads escapes."
  [{:keys [state record?]} v how]
  (when record?
    (doseq [n (:reads v) :when (integer? n)]
      (swap! state update-in [:escapes n] (fnil conj #{}) how)))
  v)

(defn- walk-args
  "The arguments of a call: each one a value the call computes with."
  [st forms how]
  (apply reads (map #(escape! st (walk st %) how) forms)))

(defn- record-site!
  "Record the site of `(f args…)` and return its value."
  [{:keys [state] :as st} kind form args]
  (let [[dist-form & more] args
        [value-form opts] (case kind
                            :observed [(first more) (rest more)]
                            [nil more])
        opts (options opts)
        kind (if (= kind :choose) (if (contains? opts :observe) :observed :latent) kind)
        value-form (if (and (= kind :observed) (nil? value-form)) (:observe opts) value-form)
        ;; the distribution: a constructor of foerster.dist we can read the
        ;; arguments of, or an expression
        [ctor dist-args] (when (and (seq? dist-form) (symbol? (first dist-form)))
                           (let [q (resolve-sym (:ns st) (:locals st) (first dist-form))]
                             (when q [q (rest dist-form)])))
        ctor (when (and ctor (= dist-ns (namespace ctor))) (symbol (name ctor)))
        arg-values (if ctor
                     (mapv (fn [a] (assoc (walk st a) :form a)) dist-args)
                     [(assoc (escape! st (walk st dist-form) :distribution) :form dist-form)])
        value (when value-form (escape! st (walk st value-form) :observed-value))
        id-form (:id opts)
        id-value (when id-form (walk st id-form))
        n (count (:sites @state))
        site {:kind kind
              :address (cond (:known? id-value) (:value id-value)
                             id-form id-form
                             :else [:loc (:line (meta form)) (:column (meta form))])
              :structural? (nil? id-form)
              ;; where it is written, as a site's runtime payload has it
              :loc (when (:line (meta form))
                     [(str (ns-name (:ns st))) (:line (meta form)) (:column (meta form))])
              :dist (cond-> {:args arg-values}
                      ctor (assoc :ctor ctor)
                      (not ctor) (assoc :expr dist-form))
              :reads (reduce set/union (or (:reads value) #{}) (map :reads arg-values))
              ;; an observed site's value, when the walk knows it
              :observed (when (:known? value) {:value (:value value)})
              :address-reads (or (:reads id-value) #{})
              :control (:control st)
              ;; the tests of the enclosing ifs a run must pass, [expr branch]
              :path (:path st)
              :loops (mapv :id (:loops st))
              ;; the iteration of each enclosing unrolled loop
              :iterations (into {} (keep (fn [{:keys [id iter]}] (when iter [id iter]))) (:loops st))}]
    (when-let [probe (:probe st)]
      (swap! probe conj (set/union (:reads site) (:control site))))
    (when (:record? st)
      (swap! state update :sites assoc n site)
      (doseq [{:keys [id]} (:loops st)]
        (swap! state update-in [:loops id :sites] (fnil conj #{}) n)))
    (case kind
      :latent {:reads #{n} :exact n :expr {:site n}}
      :observed (or value const)
      const)))

(def unroll-budget
  "Iterations a sequential loop with known control is unrolled up to."
  1000)

(defn- bind [st syms vals frame record?]
  (-> st
      (assoc :record? record?)
      (update :env into (map vector syms vals))
      (update :loops conj frame)))

(defn- sequential?
  "Whether a site in the body of a loop reads a value one of its variables
  carries from a site of an earlier iteration. `vals` are the variables'
  fixpoint values; sites met in passes that do not record are numbered
  `n0` and up."
  [st syms vals body n0]
  (let [tag (gensym "probe")
        probe (atom [])
        carrying (set (keep (fn [[sym v]] (when (some #(and (integer? %) (>= % n0)) (:reads v)) sym))
                            (map vector syms vals)))
        marked (mapv (fn [v sym] (update v :reads conj [:via tag sym])) vals syms)]
    (walk (-> (bind st syms marked {:id nil :recurs (atom [])} false) (assoc :probe probe))
          (cons 'do body))
    (boolean (some (fn [rs] (some #(and (vector? %) (= tag (second %)) (contains? carrying (nth % 2))) rs))
                   @probe))))

(defn- iterations
  "The loop variables' values of each iteration of a loop whose control is
  known (each recur decided by known tests), or nil past `unroll-budget`."
  [st syms inits body]
  (loop [vals inits acc []]
    (when (<= (count acc) unroll-budget)
      (let [recurs (atom [])]
        (walk (-> (bind st syms vals {:id nil :recurs recurs} false) (dissoc :probe)) (cons 'do body))
        (let [rs @recurs]
          (cond (empty? rs) (conj acc vals)
                (or (next rs) (:uncertain? (first rs))) nil
                :else (recur (:vals (first rs)) (conj acc vals))))))))

(defn- walk-loop
  "loop*. A sequential loop whose control is known (`specialize`) is unrolled:
  its body is walked once per iteration with the variables' values, so its
  sites get concrete addresses and iteration-level edges. Otherwise iterate
  the body until what the loop variables read is stable, then walk it once
  more recording sites; in that walk each loop variable also reads a marker
  [:via loop sym], so a site that reads a value carried from an earlier
  iteration can be told from one that reads its own iteration's."
  [{:keys [state] :as st} [_ bvec & body]]
  (let [syms (vec (binding-syms bvec))
        record? (:record? st)
        n0 (count (:sites @state))
        [st inits] (reduce (fn [[st acc] [s init]]
                             (let [v (walk st init)]
                               [(-> st (update :env assoc s v) (update :locals conj s)) (conj acc v)]))
                           [st []] (partition 2 bvec))
        body-pass (fn [vals record? frame]
                    (let [recurs (atom [])
                          result (walk (bind st syms vals (assoc frame :recurs recurs) record?) (cons 'do body))]
                      [result @recurs]))
        vals (loop [vals inits k 0]
               (let [[_ recurs] (body-pass vals false {:id nil})
                     vals' (mapv (fn [v i] (apply reads v (map #(nth (:vals %) i const) recurs)))
                                 vals (range (count syms)))]
                 (cond
                   (= (map :reads vals') (map :reads vals)) vals'
                   ;; no fixpoint: what the loop carries is not known, so
                   ;; nothing about it may be relied on
                   (> k 20) (do (when (:record? st)
                                  (swap! state update :unknown conj {:reason :loop-not-converged :bindings syms}))
                                vals')
                   :else (recur vals' (inc k)))))
        unrolled (when (and record? (every? :known? inits)
                            (or (:unroll-plates? st) (sequential? st syms vals body n0)))
                   (iterations st syms inits body))
        id (when record? (let [id (count (:loops @state))]
                           (swap! state assoc-in [:loops id] {:bindings syms :sites #{}})
                           id))]
    (cond
      unrolled
      ;; the dry run decided the iterations; the recorded walk threads its
      ;; own recur values, whose sites are the recorded ones
      (let [result (loop [vals inits k 0]
                     (let [[result recurs] (body-pass vals true {:id id :iter k})]
                       (if (seq recurs) (recur (:vals (first recurs)) (inc k)) result)))]
        (swap! state update-in [:loops id] assoc :unrolled (count unrolled))
        result)

      record?
      (let [marked (mapv (fn [v s] (-> v (update :reads conj [:via id s]) (assoc :expr s))) vals syms)
            [result _] (body-pass marked true {:id id})]
        (swap! state assoc-in [:loops id :carried] (zipmap syms (map :reads vals)))
        result)

      :else (first (body-pass vals false {:id nil})))))

(defn- walk-call [{:keys [ns locals env] :as st} form]
  (let [[f & args] form
        q (resolve-sym ns locals f)]
    (cond
      (contains? site-fns q) (record-site! st (site-fns q) form args)

      (contains? spin-macros q) (walk st (cons 'do args))

      (macro? ns locals f)
      (walk st (binding [*ns* ns] (macroexpand-1 form)))

      (or (contains? pure-fns q) (static-math? f))
      (let [vs (mapv #(walk st %) args)
            exprs (map :expr vs)
            expr (when (every? some? exprs) (cons (or q f) exprs))
            r (if (every? :known? vs) (invoke q f (map :value vs)) ::fail)]
        (if (= ::fail r)
          (assoc (apply reads vs) :expr expr)
          (known r expr)))

      :else
      (reads (if (symbol? f) (get env f const) (walk st f))
             (walk-args st args {:call f})))))

(defn- contains-site? [{:keys [ns locals]} form]
  (some (fn [x] (and (seq? x) (symbol? (first x))
                     (contains? site-fns (resolve-sym ns locals (first x)))))
        (tree-seq coll? seq form)))

(defn walk
  "The value `form` computes, recording the sites it runs into `(:state st)`."
  [{:keys [env state] :as st} form]
  (cond
    ;; a name not bound in the model: an input or a global
    (symbol? form) (get env form {:reads #{} :exact nil :expr form})

    (and (seq? form) (seq form))
    (let [[op & args] form]
      (case op
        quote (known (second form) form)
        var const
        do (if (seq args) (do (doseq [a (butlast args)] (walk st a)) (walk st (last args))) const)
        let* (let [[bvec & body] args
                   st' (reduce (fn [st [s init]]
                                 (-> st (update :env assoc s (walk st init)) (update :locals conj s)))
                               st (partition 2 bvec))]
               (walk st' (cons 'do body)))
        loop* (walk-loop st form)
        ;; in an unrolled loop a recur value only binds the next iteration's
        ;; variable, which the walk follows; elsewhere it is carried unseen
        recur (let [unrolled? (:iter (peek (:loops st)))
                    vals (mapv #(let [v (walk st %)] (if unrolled? v (escape! st v :loop-carry))) args)]
                (when-let [{:keys [recurs]} (peek (:loops st))]
                  (swap! recurs conj {:vals vals :uncertain? (:uncertain? st)}))
                const)
        if (let [[test then else] args
                 t (walk st test)]
             (if (:known? t)
               ;; a known test: only the branch taken runs
               (walk st (if (:value t) then else))
               (let [st' (-> st (update :control set/union (:reads t)) (assoc :uncertain? true))
                     a (walk (update st' :path conj [(:expr t) true]) then)
                     b (walk (update st' :path conj [(:expr t) false]) else)]
                 (assoc (reads t a b)
                        :expr (when (and (:expr t) (:expr a) (or (nil? else) (:expr b)))
                                (list 'if (:expr t) (:expr a) (:expr b)))))))
        case* (let [[e _shift _mask default imap] (rest form)
                    t (walk st (second form))
                    st' (-> st (update :control set/union (:reads t)) (assoc :uncertain? true))]
                (apply reads t (walk st' default) (map #(walk st' (second %)) (vals imap))))
        fn* (let [captured (keep env (filter symbol? (tree-seq coll? seq form)))]
              (when (contains-site? st form)
                (swap! state update :unknown conj {:reason :closure :form form}))
              (doseq [v captured] (escape! st v :closure))
              ;; a closure reads what it closes over
              (apply reads captured))
        letfn* (do (when (contains-site? st form)
                     (swap! state update :unknown conj {:reason :closure :form form}))
                   (walk st (cons 'do (rest args))))
        ;; a caught exception continues the run on a path that depends on
        ;; values the walk does not follow
        try (do (when (and (:record? st)
                           (some #(and (seq? %) (= 'catch (first %))) args))
                  (swap! state update :unknown conj {:reason :try-catch :form form}))
                (walk-args st args :host))
        (catch finally throw new set! . monitor-enter monitor-exit)
        (walk-args st args :host)
        (walk-call st form)))

    ;; a collection holds its elements: where they go is where it goes
    (map? form) (let [ks (mapv #(walk st %) (keys form))
                      vs (mapv #(walk st %) (vals form))]
                  (cond-> (assoc (apply reads (concat ks vs))
                                 :expr (when (every? :expr (concat ks vs))
                                         (zipmap (map :expr ks) (map :expr vs))))
                    (every? :known? (concat ks vs))
                    (assoc :known? true :value (zipmap (map :value ks) (map :value vs)))))
    (coll? form) (let [vs (mapv #(walk st %) form)
                       rebuild (fn [xs] (cond (vector? form) (vec xs) (set? form) (set xs) :else nil))]
                   (cond-> (assoc (apply reads vs) :expr (when (every? :expr vs) (rebuild (map :expr vs))))
                     (and (every? :known? vs) (or (vector? form) (set? form)))
                     (assoc :known? true :value (rebuild (map :value vs)))))
    :else (known form form)))

;; -----------------------------------------------------------------------------
;; Graph and facts
;; -----------------------------------------------------------------------------

(defn analyze-form
  "The site graph of a model's body `form` (see the namespace doc), symbols
  resolved in `ns` (default `*ns*`). `locals` are the model's inputs: names
  bound outside the form, which read no site; `known` {sym value} those
  whose values are given (`specialize`)."
  ([form] (analyze-form form nil))
  ([form {:keys [ns locals unroll-plates?] given :known :or {ns *ns*}}]
   (let [state (atom {:sites {} :loops {} :unknown [] :escapes {}})
         st {:ns (the-ns ns) :locals (into (set locals) (keys given))
             :env (into {} (for [[sym v] given] [sym (known v sym)]))
             :control #{} :path [] :loops [] :state state :record? true
             :unroll-plates? unroll-plates?}]
     (let [result (walk st form)
           {:keys [sites loops unknown escapes]} @state
           marker? vector?
           strip (fn [xs] (into #{} (remove marker?) xs))]
       {:sites (into {} (for [[n s] sites]
                          [n (-> s
                                 (update :reads strip)
                                 (update :control strip)
                                 (update :address-reads strip)
                                 ;; a known argument keeps its value if it is a scalar
                                 (update-in [:dist :args]
                                            (fn [as] (mapv #(cond-> (update % :reads strip)
                                                              (not (scalar? (:value %))) (dissoc :value))
                                                           as)))
                                 (cond-> (escapes n) (assoc :escapes (escapes n))))]))
        :loops (into {} (for [[id l] loops]
                          (let [;; the loop's sites whose values a site of the loop
                                ;; reads in a later iteration
                                carried (if (:unrolled l)
                                          (into #{}
                                                (for [n (:sites l)
                                                      m (concat (:reads (sites n)) (:control (sites n)))
                                                      :when (and (integer? m) (contains? (:sites l) m)
                                                                 (< (get-in sites [m :iterations id])
                                                                    (get-in sites [n :iterations id])))]
                                                  m))
                                          (into #{}
                                                (for [n (:sites l)
                                                      m (concat (:reads (sites n)) (:control (sites n)))
                                                      :when (and (marker? m) (= id (second m)))
                                                      c (set/intersection (get (:carried l) (nth m 2)) (:sites l))]
                                                  c)))]
                            [id (cond-> {:bindings (:bindings l)
                                         :sites (:sites l)
                                         :carried carried
                                         :kind (if (seq carried) :sequential :plate)}
                                  (:unrolled l) (assoc :unrolled (:unrolled l)))])))
        :unknown unknown
        ;; the model's value: known, or its expression of the sites
        :result (cond (:known? result) {:value (:value result)}
                      (some? (:expr result)) {:expr (:expr result)})}))))

(defmacro analyze
  "The site graph of the model `body` (a `spin` form or the forms inside one),
  analyzed where it is written. Locals in scope are its inputs."
  [& body]
  `(quote ~(analyze-form (if (next body) (cons 'do body) (first body))
                         {:ns *ns* :locals (keys &env)})))

(defonce ^:private applications
  ;; spin -> [model args], weakly: what `applied` answers
  (java.util.Collections/synchronizedMap (java.util.WeakHashMap.)))

(defn applied
  "[model args] for a spin a `model` made, or nil."
  [s]
  (.get ^java.util.Map applications s))

(defmacro model
  "`(fn args (spin body…))`, carrying the site graph of `body` as metadata
  (`graph-of`): a model whose structure inference can read. The spins it
  makes remember it and their arguments (`applied`)."
  [args & body]
  (let [g (analyze-form (cons 'do body) {:ns *ns* :locals (concat (keys &env) (flatten args))})]
    `(let [self# (promise)
           make# (fn ~args (org.replikativ.spindel.spin.cps/spin ~@body))
           f# (with-meta (fn [& xs#]
                           (let [s# (apply make# xs#)]
                             (.put ^java.util.Map @#'applications s# [@self# (vec xs#)])
                             s#))
                {::graph '~g
                 ::source {:ns '~(ns-name *ns*) :params '~args :body '~body}})]
       (deliver self# f#)
       f#)))

(defn graph-of
  "The site graph a `model` carries, or nil."
  [f]
  (::graph (meta f)))

(defn specialize
  "The site graph of the `model` applied to `args`: its inputs are known, so
  calls of `pure-fns` on known values are evaluated, an `if` with a known test
  walks only the branch taken, and a sequential loop whose control becomes
  known is unrolled (up to `unroll-budget` iterations) — its sites get
  concrete addresses, edges between iterations and `:iterations`. Plates stay
  symbolic. The model's parameters must be plain symbols."
  [model & args]
  (let [{:keys [ns params body]} (::source (meta model))]
    (when-not (every? symbol? params)
      (throw (ex-info "specialize needs a model whose parameters are symbols"
                      {:type ::destructured-params :params params})))
    (analyze-form (cons 'do body) {:ns ns :known (zipmap params args)})))

(defn instance
  "As `specialize`, but plates with known control are unrolled too: the
  model's instance graph for its data, one site per site a run meets. Its
  size grows with the data; for compiling a density (`foerster.compiled`)."
  [model & args]
  (let [{:keys [ns params body]} (::source (meta model))]
    (when-not (every? symbol? params)
      (throw (ex-info "instance needs a model whose parameters are symbols"
                      {:type ::destructured-params :params params})))
    (analyze-form (cons 'do body) {:ns ns :known (zipmap params args) :unroll-plates? true})))

(defn children
  "The sites that read site `n`, or run only depending on it."
  [sites n]
  (into #{} (keep (fn [[m s]] (when (or (contains? (:reads s) n) (contains? (:control s) n)) m))) sites))

;; -----------------------------------------------------------------------------
;; Link functions
;; -----------------------------------------------------------------------------

(defn- mentions? [e n]
  (boolean (some #(= {:site n} %) (tree-seq coll? seq e))))

(defn- plus [& xs]
  (let [k (reduce + 0 (filter number? xs))
        xs (cond-> (vec (remove number? xs)) (not (zero? k)) (conj k))]
    (case (count xs) 0 0 1 (first xs) (cons 'clojure.core/+ xs))))

(defn- times [& xs]
  (let [k (reduce * 1 (filter number? xs))
        xs (remove number? xs)]
    (cond (zero? k) 0
          (empty? xs) k
          :else (let [xs (cond->> xs (not (== 1 k)) (cons k))]
                  (if (next xs) (cons 'clojure.core/* xs) (first xs))))))

(defn linear
  "{:const a :coefs {n b}} with `e` = a + Σ b·xₙ over the sites `ns` it
  mentions, xₙ the value of site n and a and every b expressions mentioning
  none of `ns`; nil if `e` is not linear in them."
  [e ns]
  (let [dependent? (fn [x] (some #(mentions? x %) ns))
        scale (fn [k {:keys [const coefs]}]
                {:const (times k const) :coefs (update-vals coefs #(times k %))})
        sum (fn [parts]
              {:const (apply plus (map :const parts))
               :coefs (or (apply merge-with plus (map :coefs parts)) {})})]
    (cond
      (and (map? e) (contains? ns (:site e))) {:const 0 :coefs {(:site e) 1}}
      (not (dependent? e)) {:const e :coefs {}}
      (seq? e)
      (let [[op & xs] e
            parts (map #(linear % ns) xs)]
        (when (every? some? parts)
          (case op
            clojure.core/+ (sum parts)
            clojure.core/- (if (next parts)
                             (sum (cons (first parts) (map #(scale -1 %) (rest parts))))
                             (scale -1 (first parts)))
            clojure.core/* (let [dependent (filter dependent? xs)]
                             (when (= 1 (count dependent))
                               (scale (apply times (remove dependent? xs))
                                      (linear (first dependent) ns))))
            clojure.core// (let [[x & ds] xs]
                             (when (and (seq ds) (not-any? dependent? ds))
                               (let [d (apply times ds)
                                     {:keys [const coefs]} (linear x ns)]
                                 {:const (list 'clojure.core// const d)
                                  :coefs (update-vals coefs #(list 'clojure.core// % d))})))
            nil)))
      :else nil)))

(defn affine
  "[a b] with `e` = a + b·x, x the value of site `n` and a, b expressions
  not mentioning it; nil if `e` is not affine in it."
  [e n]
  (when-let [{:keys [const coefs]} (linear e #{n})]
    [const (get coefs n 0)]))

(defn- link
  "How the child's parameter `arg` depends on the prior site `n` in family
  table `table`'s slot for constructor `ctor`: {:family f} when it is `n`
  itself, {:family :normal-affine :a a :b b} for a Normal mean a + b·n."
  [prior-ctor ctor slot arg n]
  (cond
    (= n (:exact arg)) {:family (second (get-in conjugate-families [prior-ctor ctor]))}
    (and (= 'normal prior-ctor) (= 'normal ctor) (= 0 slot))
    (when-let [[a b] (some-> (:expr arg) (affine n))]
      (when-not (and (number? b) (zero? b))
        {:family :normal-affine :a a :b b}))))

(defn- conjugate-pair
  "The conjugate family the latent site `n` forms with all its children, if it
  does: every child is a constructor of the family's likelihood whose
  parameter argument is `n` itself (or, for a Normal mean, affine in `n`),
  reads `n` nowhere else, no site's existence depends on `n`, and `n`'s value
  goes nowhere else but the model's result. `:links` gives each child's
  affine link; `:given` the other sites the links read."
  [sites n]
  (let [{:keys [dist escapes]} (sites n)
        prior (:ctor dist)
        table (get conjugate-families prior)
        kids (children sites n)]
    (when (and table (seq kids) (empty? escapes))
      (let [links
            (for [k kids
                  :let [{:keys [dist control address-reads]} (sites k)
                        [slot] (get table (:ctor dist))
                        arg (when slot (nth (:args dist) slot nil))]]
              (when (and slot arg
                         (= :observed (:kind (sites k)))
                         (not (contains? control n))
                         (not (contains? address-reads n))
                         (every? (fn [[i a]] (or (= i slot) (not (contains? (:reads a) n))))
                                 (map-indexed vector (:args dist))))
                (some-> (link prior (:ctor dist) slot arg n) (assoc :child k))))
            family (cond (some nil? links) nil
                         (every? #(= :normal-mean (:family %)) links) :normal-mean
                         (every? #(#{:normal-mean :normal-affine} (:family %)) links) :normal-affine
                         (apply = (map :family links)) (:family (first links)))]
        (when family
          (cond-> {:family family
                   :children kids
                   :implemented? (contains? implemented-families family)}
            (= :normal-affine family)
            (assoc :links (into {} (for [{:keys [child a b family]} links]
                                     [child (if (= :normal-mean family) {:a 0 :b 1} {:a a :b b})]))
                   :given (into #{} (for [{:keys [a b]} links
                                          e [a b]
                                          x (tree-seq coll? seq e)
                                          :when (and (map? x) (:site x))]
                                      (:site x))))))))))

(defn- linear-gaussian-blocks
  "Sets of Normal latent sites whose joint posterior is Gaussian given the
  rest: every site's mean is linear in the set and its scale reads none of
  it, every child of the set is in it or a Normal observation of that form,
  no site's existence depends on the set, and no value of it escapes. Each
  block is a connected component, with its observations."
  [sites]
  (let [controlling (reduce set/union #{} (map :control (vals sites)))
        normal? (fn [s] (= 'normal (get-in s [:dist :ctor])))
        admissible? (fn [block n]
                      (let [{:keys [dist address-reads]} (sites n)
                            [mean sd] (:args dist)]
                        (and (normal? (sites n))
                             (some? (:expr mean))
                             (linear (:expr mean) block)
                             (empty? (set/intersection (:reads sd) block))
                             (empty? (set/intersection address-reads block)))))
        block (loop [block (into #{} (for [[n s] sites
                                           :when (and (= :latent (:kind s)) (normal? s)
                                                      (empty? (:escapes s))
                                                      (not (contains? controlling n)))]
                                       n))]
                (let [ok? (fn [n]
                            (and (admissible? block n)
                                 (every? #(or (contains? block %)
                                              (and (= :observed (:kind (sites %)))
                                                   (admissible? block %)
                                                   (empty? (set/intersection (:control (sites %)) block))))
                                         (children sites n))))
                      block' (into #{} (filter ok?) block)]
                  (if (= block block') block (recur block'))))
        observations (into #{} (for [n block c (children sites n)
                                     :when (= :observed (:kind (sites c)))]
                                 c))
        ;; connected components over latent-latent and shared-observation edges
        neighbours (fn [n] (into #{} (filter block)
                                 (concat (:reads (sites n)) (children sites n)
                                         (for [c (children sites n)
                                               :when (contains? observations c)
                                               p (:reads (sites c))]
                                           p))))]
    (loop [todo block out []]
      (if-let [n (first todo)]
        (let [component (loop [seen #{n} frontier [n]]
                          (if-let [m (first frontier)]
                            (let [new (remove seen (neighbours m))]
                              (recur (into seen new) (into (subvec frontier 1) new)))
                            seen))]
          (recur (reduce disj todo component)
                 (conj out {:latents component
                            :observations (into #{} (filter #(seq (set/intersection (:reads (sites %)) component)))
                                                observations)})))
        out))))

(defn facts
  "Facts about the model the graph `g` describes, as a vector of maps."
  [{:keys [sites loops unknown]}]
  (let [ctor-support (fn [s] (get support (get-in s [:dist :ctor])))
        latent (filter (fn [[_ s]] (= :latent (:kind s))) sites)
        controlling (reduce set/union #{} (map :control (vals sites)))]
    (vec
     (concat
      (for [[n s] latent
            :let [sup (ctor-support s)]]
        (cond-> {:fact :latent :site n :address (:address s) :support (or sup :unknown)
                 :decides-control? (contains? controlling n)}
          (:escapes s) (assoc :escapes (:escapes s))))
      (for [[n s] latent :when (= :finite (ctor-support s))]
        {:fact :finite-support :site n :address (:address s)})
      (for [[n s] latent
            :let [sup (ctor-support s)]
            :when (and (continuous sup) (not (contains? controlling n)))]
        {:fact :gradient-candidate :site n :address (:address s)})
      (for [[id l] loops :when (seq (:sites l))]
        {:fact (:kind l) :loop id :over (:bindings l) :sites (:sites l)})
      (for [[n s] latent
            :let [pair (conjugate-pair sites n)]
            :when pair]
        (assoc pair :fact :conjugate :prior n :address (:address s)))
      (for [b (linear-gaussian-blocks sites)
            :when (seq (:observations b))]
        (assoc b :fact :linear-gaussian))
      (for [[n s] latent
            :let [kids (children sites n)
                  parents (set/union (:reads s) (:control s))
                  co (reduce set/union #{} (map #(set/union (:reads (sites %)) (:control (sites %))) kids))]]
        {:fact :markov-blanket :site n
         :blanket (disj (set/union parents kids co) n)})
      (for [u unknown] (assoc u :fact :unknown))))))

(defn explain
  "A readable account of the graph `g` and its facts."
  [g]
  (let [{:keys [sites]} g
        addr #(pr-str (:address (sites %)))
        addrs #(str/join ", " (map addr (sort %)))]
    (str/join
     "\n"
     (concat
      ["Sites:"]
      (for [[n s] (sort-by key sites)]
        (format "  %-14s %-8s %-18s reads %s%s"
                (pr-str (:address s)) (name (:kind s))
                (or (some-> s :dist :ctor str) (pr-str (get-in s [:dist :expr])))
                (if (seq (:reads s)) (addrs (:reads s)) "—")
                (if (seq (:control s)) (str "; runs if " (addrs (:control s))) "")))
      ["Facts:"]
      (for [f (facts g)
            :when (not= :latent (:fact f))]
        (case (:fact f)
          :finite-support (str "  finite support: " (addr (:site f)))
          :gradient-candidate (str "  continuous, decides no branch: " (addr (:site f)))
          :plate (str "  plate over " (pr-str (:over f)) ": " (addrs (:sites f)))
          :sequential (str "  sequential loop over " (pr-str (:over f)) ": " (addrs (:sites f)))
          :conjugate (str "  conjugate " (name (:family f)) ": " (addr (:prior f))
                          " with " (addrs (:children f))
                          (when-not (:implemented? f) " (family not in foerster.conjugate)"))
          :linear-gaussian (str "  linear-Gaussian block: " (addrs (:latents f))
                                " observed by " (addrs (:observations f)))
          :markov-blanket (str "  Markov blanket of " (addr (:site f)) ": "
                               (if (seq (:blanket f)) (addrs (:blanket f)) "—"))
          :unknown (str "  not analyzed (" (name (:reason f)) "): " (pr-str (:form f)))))))))
