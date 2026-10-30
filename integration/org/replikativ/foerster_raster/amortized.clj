(ns org.replikativ.foerster-raster.amortized
  "Optional CPU GSDM backend for Foerster amortized tasks. Staged here so core
  remains dependency-free of Raster. Only public Raster compilation APIs;
  compiled engines may be retained explicitly by the caller, never cached here."
  (:require [org.replikativ.foerster.amortized :as amortized]
            [raster.dl.gsdm :as g]
            [raster.dl.diffusion :as diffusion]
            [raster.params :as params]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- layout [task]
  (let [types (vec (distinct (map :type (:schema task))))
        slots (vec (mapcat (fn [v] (repeat (:width v) {:name (:name v) :role (:role v) :type (:type v)})) (:schema task)))
        indices (reduce-kv (fn [out i slot] (update out (:name slot) (fnil conj []) i)) {} slots)
        edges (if (nil? (:attention-edges task))
                (vec (for [i (range (count slots)) j (range (count slots))] [i j]))
                ;; Expand only declared neighbors, rather than scanning D² pairs
                ;; to discover an O(D) sparse graph. Sorting preserves checkpoint layout.
                (vec (sort (for [[a b] (set (concat (map #(vector % %) (keys indices))
                                                    (:attention-edges task)
                                                    (map (comp vec reverse) (:attention-edges task))))
                                 i (indices a) j (indices b)] [i j]))))]
    {:dimension (count slots) :slots slots :n-spaces (count types)
     :spaces (long-array (map #(.indexOf types (:type %)) slots))
     :src (long-array (map first edges)) :dst (long-array (map second edges))
     :n-edges (count edges)}))

(defn prepare
  "Compile architecture once. A supplied :compiled engine must have matching
  :dimension and config, and :forward/:training functions. CPU/double only;
  GPU qualification remains separate work. No training examples are needed."
  [task {:keys [emb-dim n-layers n-heads timesteps prediction compiled target-device target dtype]
         :or {emb-dim 16 n-layers 2 n-heads 2 timesteps 64 prediction :v}}]
  (when (or target-device target (and dtype (not= :double dtype)))
    (throw (ex-info "This GSDM adapter supports CPU/JVM-AOT double precision only"
                    {:target-device target-device :target target :dtype dtype})))
  (when-not (and (pos-int? emb-dim) (pos-int? n-layers) (pos-int? n-heads)
                 (zero? (mod emb-dim n-heads)) (integer? timesteps) (> timesteps 1)
                 (#{:v :x0} prediction))
    (throw (ex-info "Invalid GSDM architecture or prediction parameterization" {})))
  (let [started (System/nanoTime) layout (layout task)
        config (assoc (g/make-gsdm-config :emb-dim emb-dim :n-layers n-layers :n-heads n-heads
                                          :n-spaces (:n-spaces layout) :timesteps timesteps) :prediction prediction)
        shape-keys [:emb-dim :n-layers :n-heads :n-spaces :n-states]
        _ (when (and compiled
                     (not (and (= (:dimension layout) (:dimension compiled))
                               (= (select-keys config shape-keys) (select-keys (:config compiled) shape-keys))
                               (= (mapv vec [(:src layout) (:dst layout) (:spaces layout)])
                                  (when (:layout compiled)
                                    (mapv vec [(get-in compiled [:layout :src])
                                               (get-in compiled [:layout :dst])
                                               (get-in compiled [:layout :spaces])]))))))
            (throw (ex-info "Explicit compiled engine must match architecture AND fixed attention layout" {})))
        functions
        (or compiled
            (let [loss-var (g/make-gsdm-loss config)
                  [let-sym bindings] (g/gsdm-loss-body emb-dim n-heads (:n-spaces layout))
                  _ (when-not (= 'loss (nth bindings (- (count bindings) 2)))
                      (throw (ex-info "Raster loss generator no longer exposes the expected prediction expression" {})))
                  body (list let-sym (vec (drop-last 2 bindings)) 'pred)
                  forward-sym (symbol (str "conditional-forward-" n-layers "L-" emb-dim "d-" n-heads "h-" (:n-spaces layout) "s"))
                  forward-var (binding [*ns* (the-ns 'org.replikativ.foerster-raster.amortized)]
                                (eval (list 'raster.core/deftm forward-sym
                                            ['w ':- (g/weights-spec n-layers)
                                             'values ':- '(Array double) 'spaces ':- '(Array long)
                                             'target ':- '(Array double) 'states ':- '(Array long)
                                             'pos-emb ':- '(Array double) 'src-edges ':- '(Array long)
                                             'dst-edges ':- '(Array long) 't ':- 'Double
                                             'n-vars ':- 'Long 'n-edges ':- 'Long]
                                            ':- '(Array double) body))
                                (ns-resolve *ns* forward-sym))]
              {:forward (params/compile-aot forward-var) :training (params/compile-train-step loss-var)}))
        betas (double-array
               (for [i (range timesteps)]
                 (let [a (Math/pow (Math/cos (* (/ (+ (/ i (double timesteps)) 0.008) 1.008) (/ Math/PI 2))) 2)
                       b (Math/pow (Math/cos (* (/ (+ (/ (inc i) (double timesteps)) 0.008) 1.008) (/ Math/PI 2))) 2)]
                   (min 0.999 (max 0.0001 (- 1 (/ b a)))))))]
    (merge (select-keys functions [:forward :training])
           {:layout layout :dimension (:dimension layout) :config config :betas betas
            :bars (diffusion/compute-alphas-cumprod betas)
            :positions (g/compute-position-embeddings (:spaces layout) (:dimension layout) emb-dim)
            :preparation-seconds (/ (- (System/nanoTime) started) 1e9)})))

(defn- states [engine mask]
  (long-array (map #(if (or (= :input (:role %)) (contains? mask (:name %))) 1 0)
                   (get-in engine [:layout :slots]))))

(defn- random-states [engine task ^java.util.Random rng]
  (let [eligible (vec (map :name (filter #(#{:observation :variable} (:role %)) (:schema task))))
        mask (set (filter (fn [_] (.nextBoolean rng)) eligible))
        targets (set (map :name (filter #(#{:target :variable} (:role %)) (:schema task))))
        ;; Avoid a zero-loss fully observed update without enumerating masks.
        mask (if (every? mask targets) (disj mask (nth (vec (sort-by pr-str targets)) (.nextInt rng (count targets)))) mask)]
    (states engine mask)))

(defn- forward [payload values state t]
  (let [{:keys [engine weights]} payload {:keys [layout positions]} engine]
    ((:forward engine) weights values (:spaces layout) values state positions
                       (:src layout) (:dst layout) (double t) (long (:dimension layout)) (long (:n-edges layout)))))

(defn- normalization [engine rows]
  (let [n (double (count rows))]
    (mapv (fn [i]
            (if (= :finite (:type (get-in engine [:layout :slots i]))) [0.0 1.0]
                (let [mu (/ (reduce + (map #(nth % i) rows)) n)
                      sd (Math/sqrt (/ (reduce + (map #(Math/pow (- (nth % i) mu) 2) rows)) n))]
                  [mu (if (> sd 1e-8) sd 1.0)]))) (range (:dimension engine)))))

(defn- normalize [normalizer values]
  (double-array (map (fn [[mu sd] x] (/ (- x mu) sd)) normalizer values)))

(defn- tree-data [x]
  (cond (map? x) (update-vals x tree-data)
        (vector? x) (mapv tree-data x)
        (and x (.isArray (class x))) (vec x)
        :else x))

(defn- restore-tree! [template data]
  (cond
    (map? template)
    (do (when-not (= (set (keys template)) (set (keys data)))
          (throw (ex-info "Optimizer checkpoint keys do not match" {})))
        (doseq [[k value] template] (restore-tree! value (get data k))))
    (vector? template)
    (do (when-not (= (count template) (count data))
          (throw (ex-info "Optimizer checkpoint layers do not match" {})))
        (doseq [[a b] (map vector template data)] (restore-tree! a b)))
    :else
    (do (when-not (and (sequential? data) (= (alength ^doubles template) (count data))
                       (every? #(and (number? %) (< ##-Inf % ##Inf)) data))
          (throw (ex-info "Invalid optimizer checkpoint array" {})))
        (System/arraycopy (double-array data) 0 template 0 (count data)))))

(defn- fingerprint [x]
  (format "%064x" (java.math.BigInteger. 1
                                         (.digest (java.security.MessageDigest/getInstance "SHA-256")
                                                  (.getBytes (pr-str x) "UTF-8")))))

(defn- training-case [engine task data masks ^java.util.Random rng]
  (let [clean ^doubles (data (.nextInt rng (count data)))
        state ^longs (if masks (masks (.nextInt rng (count masks))) (random-states engine task rng))
        t (.nextInt rng (get-in engine [:config :timesteps]))
        abar (aget ^doubles (:bars engine) t) a (Math/sqrt abar) b (Math/sqrt (- 1 abar))
        n (:dimension engine) noisy (double-array n) target (double-array n)]
    (dotimes [i n]
      (let [x (aget clean i) epsilon (.nextGaussian rng)]
        (aset noisy i (if (= 1 (aget state i)) x (+ (* a x) (* b epsilon))))
        (aset target i (if (and (= :v (get-in engine [:config :prediction])) (zero? (aget state i)))
                         (- (* a epsilon) (* b x)) x))))
    {:values noisy :target target :state state :t t}))

(defn- validation-loss [payload cases]
  (when (seq cases)
    (/ (reduce + (for [{:keys [values target state t]} cases
                       :let [pred ^doubles (forward payload values state t)
                             indices (filter #(zero? (aget ^longs state %)) (range (alength pred)))]]
                   (/ (reduce + (map #(let [e (- (aget pred %) (aget ^doubles target %))] (* e e)) indices))
                      (max 1 (count indices)))))
       (count cases))))

(defn- finite-parameters? [flat]
  (every? (fn [[_ values]] (every? #(Double/isFinite (double %)) values)) flat))

(defn- validation-by-mask [payload cases]
  (into {} (for [[mask subset] (group-by
                                (fn [{:keys [state]}]
                                  (set (keep-indexed #(when (= 1 (aget ^longs state %1)) (:name %2))
                                                     (get-in payload [:engine :layout :slots])))) cases)]
             [mask {:cases (count subset) :loss (validation-loss payload subset)}])))

(defn fit
  "Fit a finite joint-data bank. Training outcomes never certify posterior
  accuracy. Validation uses fixed held-out noise/masks, best-state selection,
  and optional plateau stopping. :resume accepts a thawed backend payload."
  [task rows {:keys [updates rate seed discrete-targets stages validation-rows
                     validation-every validation-cases patience min-delta max-seconds
                     on-progress resume restore-best?]
              :or {updates 20000 rate 0.0003 seed 1 discrete-targets :argmax-relaxation
                   validation-every 1000 validation-cases 128 min-delta 0.0 restore-best? true}
              :as options}]
  (when-not (and (pos-int? updates) (number? rate) (< 0.0 rate ##Inf)
                 (pos-int? validation-every) (pos-int? validation-cases)
                 (number? min-delta) (<= 0.0 min-delta) (< min-delta ##Inf)
                 (or (nil? patience) (pos-int? patience))
                 (or (nil? max-seconds) (and (number? max-seconds) (< 0.0 max-seconds ##Inf)))
                 (or (nil? on-progress) (fn? on-progress)))
    (throw (ex-info "Invalid training budget, validation or progress settings" {})))
  (when (and (some #(and (= :finite (:type %))
                         (or (#{:target :variable} (:role %))
                             (= :any (:conditionings task))
                             (some (fn [mask] (not (contains? mask (:name %)))) (:conditionings task)))) (:schema task))
             (not= :argmax-relaxation discrete-targets))
    (throw (ex-info "Finite latent variables need explicit :discrete-targets :argmax-relaxation" {})))
  (let [started (System/nanoTime)
        stages (or stages [{:updates updates :rate rate}])
        _ (when-not (and (vector? stages) (seq stages)
                         (every? #(and (pos-int? (:updates %)) (number? (:rate %)) (< 0.0 (:rate %) ##Inf)) stages))
            (throw (ex-info "Training stages need positive update counts and finite learning rates" {})))
        budget (reduce + (map :updates stages))
        limits (vec (rest (reductions + 0 (map :updates stages))))
        rate-at (fn [step] (:rate (nth stages (count (take-while #(< % step) limits)))))
        engine (prepare task (merge (get-in resume [:engine :config]) options)) flat (g/init-gsdm-weights (:config engine))
        weights (g/flat->tree flat (get-in engine [:config :n-layers]))
        training (:training engine) adam ((:init-state training) weights)
        identity {:task (amortized/description task) :data (fingerprint rows) :seed seed
                  :config (:config engine) :validation-data (fingerprint validation-rows)
                  :validation-cases validation-cases}
        _ (when resume
            (when-not (and (:optimizer resume) (= identity (get-in resume [:optimizer :identity])))
              (throw (ex-info "Resume needs matching task, data, seed, architecture and validation protocol" {})))
            (restore-tree! flat (tree-data (:flat resume)))
            (restore-tree! adam (get-in resume [:optimizer :adam])))
        normalizer (or (:normalizer resume) (normalization engine rows))
        data (mapv #(normalize normalizer %) rows)
        held-out (mapv #(normalize normalizer %) validation-rows)
        _ (when (and patience (empty? held-out))
            (throw (ex-info "Plateau stopping requires held-out validation examples" {})))
        masks (when-not (= :any (:conditionings task)) (mapv #(states engine %) (:conditionings task)))
        cases (when (seq held-out)
                (let [rng (java.util.Random. (bit-xor (long seed) 1738937))]
                  (mapv (fn [_] (training-case engine task held-out masks rng)) (range validation-cases))))
        offset (long (or (get-in resume [:optimizer :step]) 0))
        payload {:engine engine :flat flat :weights weights :normalizer normalizer}
        elapsed #(/ (- (System/nanoTime) started) 1e9)
        snapshot (fn [step] {:flat (tree-data flat) :optimizer {:step step :identity identity :adam (tree-data adam)}})
        initial-loss (validation-loss payload cases)
        initial (snapshot offset)
        result
        (loop [done 0 first-losses [] recent [] history [] best initial best-loss initial-loss stale 0]
          (let [reason (cond
                         (and best-loss (not (Double/isFinite best-loss))) :numerical-failure
                         (and max-seconds (>= (elapsed) max-seconds)) :time-limit
                         (= done budget) :budget-exhausted
                         (and patience (>= stale patience)) :validation-plateau)]
            (if reason
              {:status reason :done done :first first-losses :recent recent :history history
               :best best :best-loss best-loss}
              (let [step (+ offset done 1)
                    ;; A step-local RNG permits exact continuation without serializing Java objects.
                    rng (java.util.Random. (unchecked-add (long seed) (unchecked-multiply (long step) 1000003)))
                    {:keys [values target state t]} (training-case engine task data masks rng)
                    loss ((:train-fn training) weights (:m adam) (:v adam)
                                               values (get-in engine [:layout :spaces]) target state (:positions engine)
                                               (get-in engine [:layout :src]) (get-in engine [:layout :dst])
                                               (double t) (long (:dimension engine)) (long (get-in engine [:layout :n-edges]))
                                               1.0 (double (rate-at (inc done))) 0.9 0.999 1e-8 (long step))
                    done (inc done)]
                (if-not (Double/isFinite loss)
                  {:status :numerical-failure :done done :first first-losses :recent recent :history history
                   :best best :best-loss best-loss :failed-step step}
                  (let [check? (or (= done budget) (zero? (mod done validation-every)))
                        vl (when check? (validation-loss payload cases))
                        finite? (and (or (nil? vl) (Double/isFinite vl))
                                     (or (not check?) (finite-parameters? flat)))
                        improved? (and vl finite? (or (nil? best-loss) (< vl (- best-loss min-delta))))
                        best (if improved? (snapshot step) best)
                        best-loss (if improved? vl best-loss)
                        event {:status :training :updates done :optimizer-step step :seconds (elapsed)
                               :training-loss loss :validation-loss vl :best-validation-loss best-loss
                               :validation-by-mask (when check? (validation-by-mask payload cases))}
                        history (if check? (conj history event) history)]
                    (when (and check? on-progress) (on-progress event))
                    (if-not finite?
                      {:status :numerical-failure :done done :first first-losses :recent recent :history history
                       :best best :best-loss best-loss :failed-step step}
                      (recur done (if (< (count first-losses) 100) (conj first-losses loss) first-losses)
                             (vec (take-last 100 (conj recent loss))) history best best-loss
                             (if (and check? vl) (if improved? 0 (inc stale)) stale)))))))))
        select-best? (or (= :numerical-failure (:status result)) (and restore-best? (seq cases)))
        selected (if select-best? (:best result) (snapshot (+ offset (:done result))))
        _ (restore-tree! flat (:flat selected))
        receipt {:status (:status result) :fit-quality (if (= :numerical-failure (:status result)) :failed :unverified)
                 :validation-scope :denoising-loss :validation-performed? (boolean (seq cases))
                 :updates (:done result) :requested-updates budget :optimizer-step (get-in selected [:optimizer :step])
                 :seed seed :stages stages :examples (count rows) :validation-examples (count validation-rows)
                 :seconds (elapsed) :first-loss (when (seq (:first result)) (/ (reduce + (:first result)) (count (:first result))))
                 :last-loss (when (seq (:recent result)) (/ (reduce + (:recent result)) (count (:recent result))))
                 :initial-validation-loss initial-loss :selected-validation-loss (validation-loss payload cases)
                 :validation-by-mask (validation-by-mask payload cases)
                 :best-validation-loss (:best-loss result) :history (:history result)
                 :selected-best? (boolean select-best?) :failed-step (:failed-step result)
                 :backend :cpu-jvm-aot :discrete-targets discrete-targets}]
    (when on-progress (on-progress receipt))
    (assoc payload :optimizer (:optimizer selected) :training receipt)))

(defn sample-rows [payload request {:keys [draws seed]}]
  (let [{:keys [engine normalizer]} payload {:keys [dimension betas bars config]} engine
        known (normalize normalizer (:values request))
        state (long-array (map #(if % 1 0) (:observed request)))
        rng (java.util.Random. seed)]
    (mapv (fn [_]
            (let [x (double-array (map-indexed (fn [i observed?] (if observed? (aget known i) (.nextGaussian rng))) (:observed request)))]
              (doseq [t (range (dec (:timesteps config)) -1 -1)]
                (let [prediction ^doubles (forward payload x state t)
                      abar (aget ^doubles bars t) beta (aget ^doubles betas t)
                      previous (if (pos? t) (aget ^doubles bars (dec t)) 1.0)]
                  (dotimes [i dimension]
                    (when (zero? (aget state i))
                      (let [z (aget x i)
                            x0 (if (= :v (:prediction config))
                                 (- (* (Math/sqrt abar) z) (* (Math/sqrt (- 1 abar)) (aget prediction i)))
                                 (aget prediction i))
                            mean (+ (* (/ (* (Math/sqrt previous) beta) (- 1 abar)) x0)
                                    (* (/ (* (Math/sqrt (- 1 beta)) (- 1 previous)) (- 1 abar)) z))]
                        (aset x i (+ mean (if (pos? t) (* (Math/sqrt beta) (.nextGaussian rng)) 0.0))))))))
              (mapv (fn [i [mu sd]] (if ((:observed request) i) ((:values request) i)
                                        (+ mu (* sd (aget x i))))) (range dimension) normalizer))) (range draws))))

(defn- freeze-payload [payload]
  {:config (get-in payload [:engine :config])
   :normalizer (:normalizer payload) :training (:training payload)
   :optimizer (:optimizer payload)
   :flat (into {} (map (fn [[k values]] [k (vec values)]) (:flat payload)))})

(defn- thaw-payload [task data options]
  (let [engine (prepare task (merge options (:config data)))
        flat (g/init-gsdm-weights (:config engine))]
    (when-not (and (= (set (keys flat)) (set (keys (:flat data))))
                   (= (:dimension engine) (count (:normalizer data)))
                   (every? #(and (= 2 (count %)) (every? (fn [x] (and (number? x) (< ##-Inf x ##Inf))) %) (pos? (second %)))
                           (:normalizer data)))
      (throw (ex-info "Invalid GSDM checkpoint layout or normalization" {})))
    (doseq [[k values] (:flat data)]
      (when-not (and (= (alength ^doubles (flat k)) (count values))
                     (every? #(and (number? %) (< ##-Inf % ##Inf)) values))
        (throw (ex-info "Invalid GSDM checkpoint parameter" {:key k})))
      (System/arraycopy (double-array values) 0 (flat k) 0 (count values)))
    (when-let [optimizer (:optimizer data)]
      (when-not (and (nat-int? (:step optimizer)) (map? (:identity optimizer)))
        (throw (ex-info "Invalid optimizer checkpoint step or identity" {})))
      (restore-tree! ((get-in engine [:training :init-state])
                      (g/flat->tree flat (get-in engine [:config :n-layers]))) (:adam optimizer)))
    {:engine engine :flat flat :weights (g/flat->tree flat (get-in engine [:config :n-layers]))
     :normalizer (:normalizer data) :training (:training data) :optimizer (:optimizer data)}))

(defn backend
  "Construct the optional GSDM backend. Compilation is explicit per training
  or restore; :compiled may reuse a caller-owned matching engine in a REPL."
  ([] (backend {}))
  ([options]
   {:id :foerster-raster/gsdm-v1
    :describe (fn [payload] {:backend :cpu-jvm-aot :dtype :double :density :unavailable
                             :config (get-in payload [:engine :config])
                             :dimension (get-in payload [:engine :dimension])
                             :attention-edges (get-in payload [:engine :layout :n-edges])
                             :training (:training payload)})
    :train (fn [task rows training-options] (spin (fit task rows (merge options training-options))))
    :sample (fn [payload request sample-options] (spin (sample-rows payload request sample-options)))
    :freeze freeze-payload :thaw (fn [task data] (thaw-payload task data options))}))
