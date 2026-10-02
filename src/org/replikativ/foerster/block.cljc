(ns org.replikativ.foerster.block
  "Numerical blocks as choice sites (the spindel side of the spindel ↔ raster
  block contract, spindel-raster doc/contract.md).

  A block is a fixed-shape group of latent variables with the density factors
  they touch, given as a description (data) and capabilities (functions over
  primitive arrays):

    (block {:block/id :gauss
            :block/latents [{:name :mu :shape [2] :support :real}]
            :block/target :complete-conditional}
           {:log-density (fn [^doubles theta inputs] lp)
            :value+grad  (fn [^doubles theta inputs] [lp ^doubles grad])})

  `(block-dist b inputs)` is the block at its inputs as a distribution whose
  density is the block's log target, so a block site is an ordinary choice
  site: `(sample (block-dist b inputs) :id :mu :init [0.0 0.0])`. Its value is
  θ, the latents flattened in declared order, as a vector of doubles.

  The target includes every factor the latents touch: their priors and the
  observations that depend on them, which must then not be observed again as
  sites of their own. A block without `:sample` cannot be drawn from; start
  it at an `:init`. `:sample` need not draw from the target (it cannot know
  its normalizer); a block drawn from under importance sampling or SMC, or
  proposed from by single-site MH, also needs `:sample-log-density`, the log
  density of what `:sample` draws, which weighs the draw:
  log target(θ) − log sample-density(θ).

  Constrained latents (`:support :positive` or `[:interval a b]`) need the
  block to declare `:block/coordinates :constrained` and give its
  capabilities in those natural coordinates (σ itself, p itself): `block`
  then works in unconstrained θ — log σ, logit of p — adding each
  transform's log-Jacobian to the density and the chain rule to the
  gradient, so HMC and NUTS move freely. θ is what the trace holds;
  `constrain` maps it back. Without the declaration, a block's capabilities
  are in unconstrained coordinates already and every latent must be `:real`."
  (:require [org.replikativ.foerster.dist :as dist]))

(def ^:private required-capabilities #{:log-density :value+grad})

(defn- size [shape] (reduce * 1 shape))

(defrecord Block [description capabilities offsets dimension])

;; Transforms: θ (unconstrained) → x (the latent), with log |dx/dθ| and its
;; derivative in θ, per coordinate.

(defn- sigmoid [t] (/ 1.0 (+ 1.0 (Math/exp (- t)))))

(defn- to-x [support t]
  (cond (= :real support) t
        (= :positive support) (Math/exp t)
        :else (let [[_ a b] support] (+ a (* (- b a) (sigmoid t))))))

(defn- to-theta [support x]
  (cond (= :real support) x
        (= :positive support) (Math/log x)
        :else (let [[_ a b] support u (/ (- x a) (- b a))] (Math/log (/ u (- 1.0 u))))))

(defn- dx-dtheta [support t]
  (cond (= :real support) 1.0
        (= :positive support) (Math/exp t)
        :else (let [[_ a b] support s (sigmoid t)] (* (- b a) s (- 1.0 s)))))

(defn- log-jacobian [support t]
  (cond (= :real support) 0.0
        (= :positive support) t
        :else (let [[_ a b] support s (sigmoid t)] (+ (Math/log (- b a)) (Math/log s) (Math/log (- 1.0 s))))))

(defn- dlog-jacobian [support t]
  (cond (= :real support) 0.0
        (= :positive support) 1.0
        :else (- 1.0 (* 2.0 (sigmoid t)))))

(defn- transformed
  "Capabilities over θ from capabilities over the constrained latents
  `supports` (one per coordinate)."
  [{:keys [log-density value+grad sample sample-log-density] :as caps} supports]
  (let [xs (fn [^doubles th] (double-array (map to-x supports th)))
        log-j (fn [^doubles th] (reduce + 0.0 (map log-jacobian supports th)))]
    (cond-> (assoc caps
                   :log-density (fn [th inputs] (+ (double (log-density (xs th) inputs)) (log-j th)))
                   :value+grad (fn [th inputs]
                                 (let [[lp g] (value+grad (xs th) inputs)]
                                   [(+ (double lp) (log-j th))
                                    (double-array (map (fn [gi su t] (+ (* gi (dx-dtheta su t)) (dlog-jacobian su t)))
                                                       (seq g) supports th))])))
      sample (assoc :sample (fn [inputs] (mapv to-theta supports (sample inputs))))
      sample-log-density (assoc :sample-log-density
                                (fn [th inputs] (+ (double (sample-log-density (xs th) inputs)) (log-j th)))))))

(defn constrain
  "The latents' values at θ: θ mapped back through each latent's transform
  (identity for `:real`), as a vector in θ's order."
  [b theta]
  (let [latents (:block/latents (:description b))
        supports (vec (mapcat (fn [l] (repeat (size (:shape l)) (:support l :real))) latents))]
    (mapv to-x supports theta)))

(defn block
  "A block from its `description` and `capabilities` (see the namespace)."
  [description capabilities]
  (let [latents (:block/latents description)
        missing (remove (set (keys capabilities)) required-capabilities)]
    (when (seq missing)
      (throw (ex-info "A block lacks required capabilities"
                      {:type ::missing-capabilities :block (:block/id description)
                       :missing (vec missing)})))
    (when-let [constrained (and (not= :constrained (:block/coordinates description))
                                (seq (remove #(= :real (:support % :real)) latents)))]
      (throw (ex-info "Constrained latents need :block/coordinates :constrained"
                      {:type ::unsupported-support :block (:block/id description)
                       :latents (mapv :name constrained)})))
    (let [sizes (mapv #(size (:shape %)) latents)
          offsets (zipmap (map :name latents) (reductions + 0 sizes))
          supports (vec (mapcat (fn [l n] (repeat n (:support l :real))) latents sizes))
          capabilities (if (= :constrained (:block/coordinates description))
                         (transformed capabilities supports)
                         capabilities)]
      (->Block description capabilities offsets (reduce + 0 sizes)))))

(defn dimension "The length of θ." [b] (:dimension b))

(defn capability
  "The capability `k` of block `b`, or nil."
  [b k]
  (get (:capabilities b) k))

(defn latent
  "The latent `name` of θ: a double for a scalar, a vector otherwise."
  [b theta name]
  (let [{:keys [shape]} (first (filter #(= name (:name %)) (:block/latents (:description b))))
        at (get (:offsets b) name)]
    (if (empty? shape)
      (nth theta at)
      (subvec (vec theta) at (+ at (size shape))))))

(defn- theta-array ^doubles [theta] (double-array theta))

(defrecord BlockDist [block inputs]
  dist/Distribution
  (-draw [_]
    (if-let [sample (capability block :sample)]
      (vec (sample inputs))
      (throw (ex-info "A block that cannot be sampled starts from an :init"
                      {:type ::no-sample :block (:block/id (:description block))}))))
  (-logpdf [_ theta]
    (if (= (dimension block) (count theta))
      ((capability block :log-density) (theta-array theta) inputs)
      ##-Inf))
  dist/DrawDensity
  (-draw-logpdf [_ theta]
    (if-let [density (capability block :sample-log-density)]
      (if (= (dimension block) (count theta))
        (double (density (theta-array theta) inputs))
        ##-Inf)
      (throw (ex-info "A block drawn from needs :sample-log-density to weigh its draws"
                      {:type ::no-sample-density :block (:block/id (:description block))})))))

(defn block-dist
  "Block `b` at `inputs`, as the distribution of its site."
  [b inputs]
  (->BlockDist b inputs))

(defn block-dist? [d] (instance? BlockDist d))

(defn value+grad
  "[log target, gradient] of `dist` (a `block-dist`) at θ, both from the
  block's `:value+grad`; the gradient as a vector."
  [dist theta]
  (let [[lp g] ((capability (:block dist) :value+grad) (theta-array theta) (:inputs dist))]
    [(double lp) (vec g)]))
