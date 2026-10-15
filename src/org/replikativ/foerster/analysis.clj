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
     :loops   {2 {:bindings [i] :sites #{1} :kind :plate | :sequential}}
     :unknown [{:reason :closure :form …}]}

  A site's value reads the sites its arguments were computed from: the
  analysis follows locals through let, loop, if and calls, over-approximating
  a call's result as reading everything its arguments read. A loop whose
  recurring values read none of its own sites runs independent iterations (a
  plate); otherwise it is sequential. What the walk cannot see — a site in a
  closure, which the spin macro cannot see either — is listed under
  `:unknown`, never guessed.

  `facts` derives from the graph: the support of each site, finite latent
  sites, plates, conjugate pairs (`foerster.conjugate`'s families, in the form
  `foerster.conjugate` would collapse) and Markov blankets. `explain` prints
  both."
  (:require [clojure.set :as set]
            [clojure.string :as str]))

;; -----------------------------------------------------------------------------
;; What the walk recognizes
;; -----------------------------------------------------------------------------

(def ^:private site-fns
  '{org.replikativ.foerster.effects/sample :latent
    org.replikativ.foerster.effects/observe :observed
    org.replikativ.foerster.effects/choose :choose
    org.replikativ.foerster.effects/factor :factor})

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

;; A value is {:reads #{site …} :exact site-or-nil}: the sites it was computed
;; from, and the site whose value it is, unchanged, if it is one.
(def ^:private const {:reads #{} :exact nil})

(defn- reads [& vs] {:reads (reduce set/union #{} (map :reads vs)) :exact nil})

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

(defn- walk-all [st forms] (apply reads (map #(walk st %) forms)))

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
                     [(assoc (walk st dist-form) :form dist-form)])
        value (when value-form (walk st value-form))
        id-form (:id opts)
        id-value (when id-form (walk st id-form))
        n (count (:sites @state))
        site {:kind kind
              :address (or id-form [:loc (:line (meta form)) (:column (meta form))])
              :structural? (nil? id-form)
              :dist (cond-> {:args arg-values}
                      ctor (assoc :ctor ctor)
                      (not ctor) (assoc :expr dist-form))
              :reads (reduce set/union (or (:reads value) #{}) (map :reads arg-values))
              :address-reads (or (:reads id-value) #{})
              :control (:control st)
              :loops (mapv :id (:loops st))}]
    (when (:record? st)
      (swap! state update :sites assoc n site)
      (doseq [{:keys [id]} (:loops st)]
        (swap! state update-in [:loops id :sites] (fnil conj #{}) n)))
    (case kind
      :latent {:reads #{n} :exact n}
      :observed (or value const)
      const)))

(defn- walk-loop
  "loop*: iterate the body until what the loop variables read is stable, then
  walk it once more recording sites. In that walk each loop variable also
  reads a marker [:via loop sym], so a site that reads a value carried from an
  earlier iteration can be told from one that reads its own iteration's."
  [{:keys [state] :as st} [_ bvec & body]]
  (let [syms (vec (binding-syms bvec))
        record? (:record? st)
        id (when record? (let [id (count (:loops @state))]
                           (swap! state assoc-in [:loops id] {:bindings syms :sites #{}})
                           id))
        [st inits] (reduce (fn [[st acc] [s init]]
                             (let [v (walk st init)]
                               [(-> st (update :env assoc s v) (update :locals conj s)) (conj acc v)]))
                           [st []] (partition 2 bvec))
        body-pass (fn [vals record?]
                    (let [recurs (atom [])
                          st' (-> st
                                  (assoc :record? record?)
                                  (update :env into (map vector syms vals))
                                  (update :loops conj {:id id :recurs recurs}))
                          result (walk st' (cons 'do body))]
                      [result @recurs]))
        vals (loop [vals inits k 0]
               (let [[_ recurs] (body-pass vals false)
                     vals' (mapv (fn [v i] (apply reads v (map #(nth % i const) recurs)))
                                 vals (range (count syms)))]
                 (if (or (= (map :reads vals') (map :reads vals)) (> k 20))
                   vals'
                   (recur vals' (inc k)))))]
    (if record?
      (let [marked (mapv (fn [v s] (update v :reads conj [:via id s])) vals syms)
            [result _] (body-pass marked true)]
        (swap! state assoc-in [:loops id :carried] (zipmap syms (map :reads vals)))
        result)
      (first (body-pass vals false)))))

(defn- walk-call [{:keys [ns locals env] :as st} form]
  (let [[f & args] form
        q (resolve-sym ns locals f)]
    (cond
      (contains? site-fns q) (record-site! st (site-fns q) form args)

      (contains? spin-macros q) (walk-all st args)

      (macro? ns locals f)
      (walk st (binding [*ns* ns] (macroexpand-1 form)))

      :else
      (reads (if (symbol? f) (get env f const) (walk st f))
             (walk-all st args)))))

(defn- contains-site? [{:keys [ns locals]} form]
  (some (fn [x] (and (seq? x) (symbol? (first x))
                     (contains? site-fns (resolve-sym ns locals (first x)))))
        (tree-seq coll? seq form)))

(defn walk
  "The value `form` computes, recording the sites it runs into `(:state st)`."
  [{:keys [env state] :as st} form]
  (cond
    (symbol? form) (get env form const)

    (and (seq? form) (seq form))
    (let [[op & args] form]
      (case op
        quote const
        var const
        do (if (seq args) (do (doseq [a (butlast args)] (walk st a)) (walk st (last args))) const)
        let* (let [[bvec & body] args
                   st' (reduce (fn [st [s init]]
                                 (-> st (update :env assoc s (walk st init)) (update :locals conj s)))
                               st (partition 2 bvec))]
               (walk st' (cons 'do body)))
        loop* (walk-loop st form)
        recur (let [vals (mapv #(walk st %) args)]
                (when-let [{:keys [recurs]} (peek (:loops st))] (swap! recurs conj vals))
                const)
        if (let [[test then else] args
                 t (walk st test)
                 st' (update st :control set/union (:reads t))]
             (reads t (walk st' then) (walk st' else)))
        case* (let [[e _shift _mask default imap] (rest form)
                    t (walk st (second form))
                    st' (update st :control set/union (:reads t))]
                (apply reads t (walk st' default) (map #(walk st' (second %)) (vals imap))))
        fn* (do (when (contains-site? st form)
                  (swap! state update :unknown conj {:reason :closure :form form}))
                ;; a closure reads what it closes over
                (apply reads (vals env)))
        letfn* (do (when (contains-site? st form)
                     (swap! state update :unknown conj {:reason :closure :form form}))
                   (walk st (cons 'do (rest args))))
        (try catch finally throw new set! . monitor-enter monitor-exit)
        (walk-all st args)
        (walk-call st form)))

    (map? form) (walk-all st (concat (keys form) (vals form)))
    (coll? form) (walk-all st form)
    :else const))

;; -----------------------------------------------------------------------------
;; Graph and facts
;; -----------------------------------------------------------------------------

(defn analyze-form
  "The site graph of a model's body `form` (see the namespace doc), symbols
  resolved in `ns` (default `*ns*`). `locals` are the model's inputs: names
  bound outside the form, which read no site."
  ([form] (analyze-form form nil))
  ([form {:keys [ns locals] :or {ns *ns*}}]
   (let [state (atom {:sites {} :loops {} :unknown []})
         st {:ns (the-ns ns) :locals (set locals) :env {} :control #{}
             :loops [] :state state :record? true}]
     (walk st form)
     (let [{:keys [sites loops unknown]} @state
           marker? vector?
           carried? (fn [id l s]
                      (some (fn [m] (and (marker? m) (= id (second m))
                                         (seq (set/intersection (get (:carried l) (nth m 2))
                                                                (:sites l)))))
                            (concat (:reads s) (:control s))))
           strip (fn [xs] (into #{} (remove marker?) xs))]
       {:sites (into {} (for [[n s] sites]
                          [n (-> s
                                 (update :reads strip)
                                 (update :control strip)
                                 (update :address-reads strip)
                                 (update-in [:dist :args] (fn [as] (mapv #(update % :reads strip) as))))]))
        :loops (into {} (for [[id l] loops]
                          [id {:bindings (:bindings l)
                               :sites (:sites l)
                               :kind (if (some #(carried? id l (sites %)) (:sites l))
                                       :sequential :plate)}]))
        :unknown unknown}))))

(defmacro analyze
  "The site graph of the model `body` (a `spin` form or the forms inside one),
  analyzed where it is written. Locals in scope are its inputs."
  [& body]
  `(quote ~(analyze-form (if (next body) (cons 'do body) (first body))
                         {:ns *ns* :locals (keys &env)})))

(defn- children [sites n]
  (into #{} (keep (fn [[m s]] (when (or (contains? (:reads s) n) (contains? (:control s) n)) m))) sites))

(defn- conjugate-pair
  "The conjugate family the latent site `n` forms with all its children, if it
  does: every child is a constructor of the family's likelihood whose
  parameter argument is exactly `n`, reads `n` nowhere else, and no site's
  existence depends on `n`."
  [sites n]
  (let [{:keys [dist]} (sites n)
        table (get conjugate-families (:ctor dist))
        kids (children sites n)]
    (when (and table (seq kids))
      (let [families
            (for [k kids
                  :let [{:keys [dist control address-reads]} (sites k)
                        [slot family] (get table (:ctor dist))]]
              (when (and slot
                         (= :observed (:kind (sites k)))
                         (not (contains? control n))
                         (not (contains? address-reads n))
                         (= n (:exact (nth (:args dist) slot nil)))
                         (every? (fn [[i a]] (or (= i slot) (not (contains? (:reads a) n))))
                                 (map-indexed vector (:args dist))))
                family))]
        (when (and (every? some? families) (apply = families))
          {:family (first families)
           :children kids
           :implemented? (contains? implemented-families (first families))})))))

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
        {:fact :latent :site n :address (:address s) :support (or sup :unknown)
         :decides-control? (contains? controlling n)})
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
          :markov-blanket (str "  Markov blanket of " (addr (:site f)) ": "
                               (if (seq (:blanket f)) (addrs (:blanket f)) "—"))
          :unknown (str "  not analyzed (" (name (:reason f)) "): " (pr-str (:form f)))))))))
