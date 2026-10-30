(ns org.replikativ.foerster.amortized
  "Backend-neutral amortized conditional sampling for declared simulator tasks.
  An artifact samples an APPROXIMATE posterior; no likelihood, importance
  correction or causal interpretation is inferred from its training loss.
  Numeric schemas and result measures are portable. Training backends are optional."
  (:refer-clojure :exclude [await])
  (:require [clojure.set :as set]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as trace]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

(defn- finite? [x] (and (number? x) (< ##-Inf x ##Inf)))
(defn- fail [message data] (throw (ex-info message (assoc data :type ::invalid-task))))
(defn- size [shape] (reduce * 1 shape))

(defn- tensor-values [value]
  (cond
    (sequential? value) value
    #?(:clj (and value (.isArray (class value))) :cljs (or (array? value) (js/ArrayBuffer.isView value)))
    #?(:clj (seq value) :cljs (array-seq value))
    :else nil))

(defn- flatten-value [shape value]
  (if (empty? shape)
    [value]
    (let [items (tensor-values value)]
      (when-not (and items (= (first shape) (count items)))
        (fail "Value does not match its fixed shape" {:shape shape :value value}))
      (vec (mapcat #(flatten-value (rest shape) %) items)))))

(defn- reshape [shape values]
  (if (empty? shape) (first values)
      (mapv #(reshape (rest shape) %) (partition (size (rest shape)) values))))

(defn- variable [v]
  (let [{:keys [name role type shape values bounds encoder]} v
        shape (or shape [])]
    (when-not (and (or (keyword? name) (vector? name)) (#{:target :observation :input :variable} role)
                   (#{:real :positive :interval :simplex :finite :embedding} type)
                   (vector? shape) (every? pos-int? shape))
      (fail "Variable needs a keyword name, role, supported type and fixed shape" {:variable v}))
    (when (and (= type :finite)
               (not (and (empty? shape) (vector? values) (seq values)
                         (= (count values) (count (set values))))))
      (fail "Finite variables need a scalar domain of distinct values" {:variable v}))
    (when (and (= type :interval)
               (not (and (= 2 (count bounds)) (every? finite? bounds) (apply < bounds))))
      (fail "Interval variables need finite increasing bounds" {:variable v}))
    (when (and (= type :simplex) (not (and (= 1 (count shape)) (> (first shape) 1))))
      (fail "A simplex needs shape [k], k > 1" {:variable v}))
    (when (and (= type :embedding)
               (not (and (#{:observation :input} role) (seq shape) (map? encoder)
                         (some? (:id encoder)) (some? (:version encoder)) (fn? (:encode encoder)))))
      (fail "Embeddings are observation/input features with an identified, versioned encoder" {:variable v}))
    (assoc v :shape shape :width (case type :finite (count values) :simplex (dec (first shape)) (size shape)))))

(defn encode
  "Encode ONE declared variable into numeric coordinates. Positive/interval
  values use log/logit; simplex uses additive log ratios in its interior;
  finite values use one-hot coordinates. Embeddings are observation features."
  [{:keys [name type shape values bounds encoder] :as variable} value]
  (if (= type :finite)
    (do (when-not (some #(= value %) values)
          (fail "Value outside finite domain" {:variable name :value value}))
        (mapv #(if (= value %) 1.0 0.0) values))
    (let [xs (flatten-value shape (if (= type :embedding) ((:encode encoder) value) value))]
      (when-not (every? finite? xs)
        (fail "Encoded values must be finite numbers" {:variable name}))
      (case type
        (:real :embedding) (mapv double xs)
        :positive (do (when-not (every? pos? xs) (fail "Positive variable outside support" {:variable name}))
                      (mapv #(Math/log %) xs))
        :interval (let [[a b] bounds]
                    (when-not (every? #(< a % b) xs) (fail "Interval encoding requires interior values" {:variable name}))
                    (mapv #(Math/log (/ (- % a) (- b %))) xs))
        :simplex (do (when-not (and (every? pos? xs) (< (Math/abs (- 1.0 (reduce + xs))) 1e-8))
                       (fail "Simplex encoding requires positive components summing to one" {:variable name}))
                     (mapv #(Math/log (/ % (peek xs))) (pop xs)))))))

(defn decode
  "Decode numeric coordinates. One-hot relaxation uses argmax: an explicitly
  approximate discrete sampler, not a categorical likelihood."
  [{:keys [name type shape values bounds width]} coordinates]
  (let [xs (vec coordinates)]
    (when-not (and (= width (count xs)) (every? finite? xs))
      (fail "Backend returned invalid coordinates" {:variable name :coordinates xs}))
    (case type
      :finite (nth values (first (apply max-key second (map-indexed vector xs))))
      :embedding (fail "Feature embeddings cannot be decoded into raw observations" {:variable name})
      :simplex (let [logits (conj xs 0.0) offset (apply max logits)
                     exps (mapv #(Math/exp (- % offset)) logits) total (reduce + exps)
                     result (mapv #(/ % total) exps)]
                 (when-not (every? pos? result) (fail "Simplex decoder underflow" {:variable name}))
                 result)
      (let [decoded (case type
                      :real xs
                      :positive (mapv #(Math/exp %) xs)
                      :interval (let [[a b] bounds]
                                  (mapv #(let [s (if (neg? %) (/ (Math/exp %) (+ 1.0 (Math/exp %)))
                                                     (/ 1.0 (+ 1.0 (Math/exp (- %)))))]
                                           (+ a (* (- b a) s))) xs)))]
        (when-not (every? finite? decoded) (fail "Decoded value overflow" {:variable name}))
        (when (and (= type :positive) (not (every? pos? decoded)))
          (fail "Positive decoder underflow" {:variable name}))
        (when (and (= type :interval) (not (every? #(< (first bounds) % (second bounds)) decoded)))
          (fail "Interval decoder rounded to a support boundary" {:variable name}))
        (reshape shape decoded)))))

(defn description
  "Serializable task contract. Encoder code, simulator and causal adapters
  are rebound by the application when restoring an artifact."
  [task]
  (-> (select-keys task [:id :schema :conditionings :attention-edges :semantics :simulate-observed?])
      (update :schema #(mapv (fn [v] (if (:encoder v) (update v :encoder dissoc :encode) v)) %))))

(defn task
  "Declare a zero-argument simulator factory returning a spin with a map result.
  :schema is an ordered vector of {:name :role :type :shape ...} descriptors.
  :conditionings declares supported sets of observed names; :input variables
  are always known. Missing observations remain latent, NOT nonexistent.
  Optional :project receives (result choices); :abduct receives (draw evidence)
  and reconstructs a reusable cf abduction. :model is a causal replay factory.
  :attention-edges are undirected hints between declared names, not causal facts.
  :simulate-observed? explicitly generates fresh values at observe sites.
  Factor-scored simulations are refused by the dataset collector."
  [simulator {:keys [id schema conditionings attention-edges] :as options}]
  (let [schema (mapv variable schema) names (mapv :name schema)
        by-name (into {} (map (juxt :name identity) schema))
        targets (set (map :name (filter #(#{:target :variable} (:role %)) schema)))
        observations (set (map :name (filter #(#{:observation :variable} (:role %)) schema)))
        conditionings (if (= :any conditionings) :any (vec (or conditionings [observations])))]
    (when-not (and (fn? simulator) (some? id) (seq schema) (= (count names) (count (set names)))
                   (some #(#{:target :variable} (:role %)) schema)
                   (or (= :any conditionings)
                       (and (seq conditionings) (= (count conditionings) (count (set conditionings)))
                            (every? #(and (set? %) (set/subset? % observations)
                                          (not (set/subset? targets %))) conditionings)))
                   (every? #(and (= 2 (count %)) (every? by-name %)) attention-edges))
      (fail "Invalid simulator, task identity, schema, conditioning masks or attention edges" {:id id}))
    (assoc options :simulator simulator :schema schema :by-name by-name
           :attention-edges (when (some? attention-edges) (mapv vec attention-edges))
           :conditionings conditionings :dimension (reduce + (map :width schema)))))

(defn encode-row
  "Encode one complete simulator row in the declared coordinate order."
  [task row]
  (when-not (map? row) (fail "Simulator projection must return a map" {}))
  (vec (mapcat (fn [v]
                 (when-not (contains? row (:name v)) (fail "Simulator row misses a variable" {:name (:name v)}))
                 (encode v (get row (:name v)))) (:schema task))))

(defn condition
  "Validate a query against the task's trained masks and encode known values.
  Returns coordinates plus a per-coordinate observed mask."
  [task evidence]
  (let [names (set (keys evidence))
        inputs (set (map :name (filter #(= :input (:role %)) (:schema task))))
        observations (set/difference names inputs)
        observable (set (map :name (filter #(#{:observation :variable} (:role %)) (:schema task))))]
    (when-not (and (map? evidence) (every? (:by-name task) names) (set/subset? inputs names)
                   (set/subset? observations observable)
                   (or (= :any (:conditionings task)) (some #(= observations %) (:conditionings task))))
      (fail "Query needs all known inputs and a declared observation mask" {:evidence names}))
    {:values (vec (mapcat #(if (contains? evidence (:name %)) (encode % (get evidence (:name %)))
                               (repeat (:width %) 0.0)) (:schema task)))
     :observed (vec (mapcat #(repeat (:width %) (contains? evidence (:name %))) (:schema task)))}))

(defn decode-targets [task coordinates targets]
  (let [targets (set targets)
        declared (set (map :name (filter #(#{:target :variable} (:role %)) (:schema task))))]
    (when-not (and (seq targets) (set/subset? targets declared) (= (:dimension task) (count coordinates)))
      (fail "Request declared targets and a complete backend coordinate vector" {:targets targets}))
    (loop [[v & more] (:schema task) offset 0 out {}]
      (if v (recur more (+ offset (:width v))
                   (if (contains? targets (:name v))
                     (assoc out (:name v) (decode v (subvec (vec coordinates) offset (+ offset (:width v))))) out))
          out))))

(defn- decode-joint [task coordinates]
  (loop [[v & more] (:schema task) offset 0 out {}]
    (if v (recur more (+ offset (:width v))
                 (if (= :embedding (:type v)) out
                     (assoc out (:name v) (decode v (subvec (vec coordinates) offset (+ offset (:width v)))))))
        out)))

(defn generative-policy
  "Prior simulation without evaluating likelihoods or honoring inference guides.
  Used only for collecting joint training examples; its ephemeral trace is NOT
  a scored posterior trace. Observe-site generation remains an explicit opt-in.
  Model hard interventions are handled by the ordinary trace policy."
  [simulate-observed?]
  (let [base (trace/policy {:simulate-observed? simulate-observed?})]
    (fn [sp old]
      (cond
        (= trace/factor-site (:savepoint/site sp))
        (fail "Factor-scored programs need an explicit normalized joint-data provider" {})

        (= trace/choose-site (:savepoint/site sp))
        (let [payload (:savepoint/payload sp) source (:dist payload)]
          (when (block/block-dist? source)
            (fail "Numerical target blocks are not normalized joint simulators; provide a generative model or joint examples" {}))
          (when (and (:observed? payload) (not simulate-observed?))
            (fail "Explicitly opt into generating observations, rather than training on fixed data" {}))
          (let [unscored (reify dist/Distribution
                           (-draw [_] (dist/draw source))
                           (-logpdf [_ _] 0.0))
                decision (base (-> sp
                                   (assoc-in [:savepoint/payload :dist] unscored)
                                   (update-in [:savepoint/payload :options] dissoc :proposal)) old)]
            (-> decision
                (update :note dissoc :log-prob :log-proposal)
                (assoc-in [:note :source-dist] source)
                (assoc-in [:note ::generative?] true))))

        :else (base sp old)))))

(defn examples
  "A spin generating n independent prior simulator rows through GFI. Sessions
  close on success and failure. Observe-site generation must be opted into;
  factors require an explicit normalized joint-data provider instead."
  [task n & [{:keys [seed] :or {seed 1}}]]
  (when-not (pos-int? n) (fail "Positive simulator count required" {:count n}))
  (spin
   (loop [i 0 out []]
     (if (= i n) out
         (let [t (await (gfi/run-policy ((:simulator task))
                                        (generative-policy (:simulate-observed? task))
                                        {:seed (+ seed i)}))
               row (try
                     (when (some #(or (= trace/factor-site (:site %))
                                      (and (get-in % [:note :observed?]) (not (:simulate-observed? task))))
                                 (trace/entries t))
                       (fail "Training requires generative rows, not fixed observations or factors" {:example i}))
                     (let [row (if-let [project (:trace-project task)] (project t)
                                       (if-let [project (:project task)]
                                         (project (:trace/result t) (trace/choices t)) (:trace/result t)))]
                       (encode-row task row)
                       row)
                     (catch #?(:clj Throwable :cljs :default) e
                       (await (gfi/close! t))
                       (throw e)))]
           (await (gfi/close! t))
           (recur (inc i) (conj out row)))))))

(defrecord ApproximateMeasure [particles diagnostics]
  m/PMeasure
  (measure-type [_] :amortized)
  (sample-measure [_ n] (m/sample-measure (m/empirical particles) n))
  (log-marginal [_] nil)
  (effective-sample-size [_] (m/effective-sample-size (m/empirical particles)))
  (measure-stats [_ f] (assoc (m/measure-stats (m/empirical particles) f) :type :amortized)))

(defn artifact
  "Bind a task to a backend payload. Backend :sample returns a spin yielding
  n full encoded rows; :freeze and :thaw convert payload to/from data.
  :train returns a spin yielding a payload. These are optional backend APIs;
  sampling never implies a usable proposal density."
  [task backend payload & [validation]]
  (when-not (and (some? (:id backend)) (fn? (:sample backend)))
    (fail "Backend needs an identity and conditional sampling capability" {}))
  (let [info (when (:describe backend) ((:describe backend) payload))
        outcome (select-keys (:training info) [:status :fit-quality :validation-performed? :validation-scope
                                               :updates :requested-updates :optimizer-step :seconds])
        assessment (cond (= :numerical-failure (:status outcome)) :numerical-failure
                         (and (= :posterior (:scope validation)) (true? (:passed? validation))) :validation-passed
                         (and (= :posterior (:scope validation)) (false? (:passed? validation))) :validation-failed
                         :else :unvalidated)
        warnings (cond-> (if (= assessment :validation-passed) []
                             [{:type assessment
                               :message "Posterior checks have not passed. Training completion and denoising loss do not certify accuracy."}])
                   (#{:budget-exhausted :time-limit :validation-plateau :numerical-failure} (:status outcome))
                   (conj {:type (:status outcome)
                          :message "Inspect the training outcome and held-out posterior checks before deployment."}))]
    {:task task :backend backend :payload payload :validation validation
     :backend-info info :training-outcome outcome :deployment-status assessment :warnings warnings
     :capabilities (cond-> #{:sample :projected-marginals}
                     (or (:abduct task) (:reconstruct task)) (conj :abduct)
                     (and (:freeze backend) (:thaw backend)) (conj :checkpoint))}))

(defn validate
  "Run an application-defined posterior validator (artifact -> spin/report).
  It must return {:scope :posterior :passed? boolean ...}. Passing applies only
  to the checks declared by that validator, not every possible observation."
  [fitted validator]
  (spin
   (let [report (await (validator fitted))]
     (when-not (and (= :posterior (:scope report)) (boolean? (:passed? report)))
       (fail "Posterior validator must declare its scope and boolean :passed?" {}))
     (artifact (:task fitted) (:backend fitted) (:payload fitted) report))))

(defn train
  "Train on generated or explicitly supplied JOINT examples, then bind an
  artifact. Backend options pass through. :simulations defaults to 4096.
  The backend may reuse these finite rows across optimizer updates."
  [task backend {:keys [examples simulations seed validation-examples validation-simulations posterior-validator]
                 :or {simulations 4096 seed 1 validation-simulations 128} :as options}]
  (when-not (fn? (:train backend)) (fail "Backend has no training capability" {}))
  (when-not (nat-int? validation-simulations) (fail "Validation simulation count must be nonnegative" {}))
  (spin
   (let [rows (or examples (await (org.replikativ.foerster.amortized/examples task simulations {:seed seed})))
         encoded (mapv #(encode-row task %) rows)
         held-out (or validation-examples
                      (when (and (nil? examples) (pos? validation-simulations))
                        (await (org.replikativ.foerster.amortized/examples task validation-simulations
                                                                           {:seed (+ seed simulations)}))))]
     (when-not (seq encoded) (fail "Training examples cannot be empty" {}))
     (let [payload (await ((:train backend) task encoded
                                            (assoc (dissoc options :examples :simulations :validation-examples :validation-simulations :posterior-validator)
                                                   :validation-rows (mapv #(encode-row task %) held-out))))]
       (let [fitted (artifact task backend payload)]
         (if posterior-validator (await (validate fitted posterior-validator)) fitted))))))

(defn resume
  "Continue training a live or thawed artifact. Reuses its simulator count and
  seed; explicit example banks must be supplied again. The backend checks data
  identity and restores optimizer state. Previously attached validation is stale
  after further training and must be performed again."
  [fitted options]
  (let [receipt (get-in fitted [:payload :training])]
    (train (:task fitted) (:backend fitted)
           (merge {:simulations (:examples receipt) :seed (:seed receipt)
                   :validation-simulations (or (:validation-examples receipt) 0)}
                  options {:resume (:payload fitted)}))))

(defn infer
  "Return a spin resolving an ApproximateMeasure over target maps. :targets
  projects samples of the trained joint, without pretending to skip neural
  computation. Equal weights are not an accuracy certificate; evidence and
  proposal densities are not estimated. Diagnostics retain this distinction."
  [artifact evidence & [{:keys [draws seed targets] :or {draws 256 seed 1}}]]
  (let [{:keys [task backend payload]} artifact
        all-targets (mapv :name (filter #(#{:target :variable} (:role %)) (:schema task)))
        targets (or targets (mapv :name (filter #(and (#{:target :variable} (:role %))
                                                      (not (contains? evidence (:name %)))) (:schema task))))
        request (condition task evidence)]
    (when (= :numerical-failure (get-in payload [:training :status]))
      (fail "Training failed numerically; inspect diagnostics and resume with corrected settings" {}))
    (when-not (and (pos-int? draws) (seq targets) (set/subset? (set targets) (set all-targets)))
      (fail "Request a positive draw count and declared targets" {:draws draws :targets targets}))
    (spin
     (let [rows (await ((:sample backend) payload request {:draws draws :seed seed}))]
       (when-not (= draws (count rows)) (fail "Backend returned the wrong draw count" {:expected draws :actual (count rows)}))
       (->ApproximateMeasure
        (mapv (fn [row]
                (when-not (= (:dimension task) (count row))
                  (fail "Backend coordinate vector has the wrong size" {}))
                (let [joint (decode-joint task row)]
                  [(assoc (m/sample-particle (select-keys joint targets) nil) ::joint-draw joint) 0.0])) rows)
        {:approximation :amortized :task (:id task) :backend (:id backend)
         :seed seed :draws draws :evidence-mask (set (keys evidence))
         :ess-meaning :equal-weight-only :validation (:validation artifact)
         :training-outcome (:training-outcome artifact) :warnings (:warnings artifact)
         :deployment-status (:deployment-status artifact)
         :backend-info (:backend-info artifact)})))))

(defn abduct
  "Reconstruct causal noise from learned draws using an explicit task adapter.
  Adapter gets (target-map evidence). No causal semantics follows from a
  learned conditional alone; it must reconstruct all noise required by replay."
  [artifact posterior evidence & [{:keys [seed] :or {seed 1}}]]
  (let [adapter (get-in artifact [:task :abduct]) reconstruct (get-in artifact [:task :reconstruct])]
    (when-not (or adapter reconstruct) (fail "This task has no causal abduction adapter" {}))
    (spin
     (loop [remaining (m/get-particles posterior) i 0 out []]
       (if (seq remaining)
         (let [[particle weight] (first remaining)
               joint (or (::joint-draw particle) (m/get-value particle))
               a (if reconstruct (await (reconstruct joint evidence {:seed (+ seed i)}))
                     (random/with-stream* seed [::abduct i] #(adapter joint evidence)))]
           (when-not (and (map? (:noise a)) (contains? a :factual) (sequential? (:unsupported a)))
             (fail "Abduction adapter must return factual values, noise and unsupported sites" {}))
           (recur (rest remaining) (inc i) (conj out (assoc a :weight weight)))) out)))))

(defn worlds
  "A spin replaying declared causal worlds using approximate abductions."
  [artifact posterior evidence specs]
  (spin
   (when-not (get-in artifact [:task :model]) (fail "Task needs a causal replay model" {}))
   (let [abductions (await (abduct artifact posterior evidence))]
     (await (cf/query ((get-in artifact [:task :model]))
                      {:abductions abductions :worlds specs})))))

(defn predictive
  "A spin generating forward rollouts from learned posterior draws. model-fn
  receives a target map and returns a simulator spin, explicitly binding the
  parameters/state used by the rollout. Observations draw fresh; no posterior
  noise is reused. This supports action evaluation, not retrospective worlds."
  [posterior model-fn n & [{:keys [seed] :or {seed 1}}]]
  (when-not (and (fn? model-fn) (pos-int? n)) (fail "Provide a forward model factory and positive rollout count" {}))
  (spin
   (let [particles (random/with-stream* seed ::predictive #(m/sample-measure posterior n))]
     (loop [remaining particles i 0 out []]
       (if (seq remaining)
         (let [[particle _] (first remaining)
               t (await (gfi/run-policy (model-fn (m/get-value particle)) (generative-policy true) {:seed (+ seed i)}))
               result {:value (:trace/result t)
                       :observations (into {} (for [entry (trace/entries t) :when (get-in entry [:note :simulated?])]
                                                [(:address entry) (:value entry)]))}]
           (await (gfi/close! t))
           (recur (rest remaining) (inc i) (conj out result))) out)))))

(defn freeze
  "Portable DATA checkpoint; no compiled closures, encoder code or GPU handles."
  [artifact]
  (when-not (contains? (:capabilities artifact) :checkpoint) (fail "Backend cannot serialize this artifact" {}))
  (let [checkpoint {:format ::artifact-v1 :task (description (:task artifact))
                    :backend (get-in artifact [:backend :id])
                    :payload ((get-in artifact [:backend :freeze]) (:payload artifact))
                    :validation (:validation artifact)}]
    (letfn [(data? [x]
              (or (nil? x) (boolean? x) (number? x) (string? x) (keyword? x) (symbol? x)
                  #?(:clj (char? x) :cljs false)
                  (and (map? x) (not (record? x)) (every? data? (keys x)) (every? data? (vals x)))
                  (and (or (vector? x) (set? x) (list? x)) (every? data? x))))]
      (when-not (data? checkpoint) (fail "Checkpoint must contain portable data, not closures, arrays or device handles" {})))
    checkpoint))

(defn thaw
  "Rebind a checkpoint to its exact task description and backend. Application
  supplies simulator/encoder/causal callbacks, including their version IDs."
  [task backend checkpoint]
  (when-not (and (= ::artifact-v1 (:format checkpoint)) (= (description task) (:task checkpoint))
                 (= (:id backend) (:backend checkpoint)) (:thaw backend))
    (fail "Artifact task, encoder identity/version or backend does not match" {}))
  (artifact task backend ((:thaw backend) task (:payload checkpoint)) (:validation checkpoint)))
