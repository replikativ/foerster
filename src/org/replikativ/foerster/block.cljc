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

  Constrained latents need the block to declare `:block/coordinates
  :constrained` and give its capabilities in those natural coordinates (σ
  itself, p itself): `block` then works in unconstrained θ, adding each
  transform's log-Jacobian to the density and the chain rule to the
  gradient, so HMC and NUTS move freely. The supports, with their
  transforms (Stan's and PyMC's):

    :real            x = θ
    :positive        x = exp θ
    [:lower a]       x = a + exp θ
    [:upper b]       x = b − exp θ
    [:interval a b]  x = a + (b − a)·logistic(θ)
    :ordered         a vector [K], increasing: x₀ = θ₀, xₖ = xₖ₋₁ + exp θₖ
    :simplex         a vector [K] of probabilities summing to 1, from K − 1
                     coordinates of θ by stick-breaking

  An unknown support, or one that does not fit its latent's shape, is
  refused when the block is defined (`::unknown-support`). (A Cholesky
  factor of a correlation matrix is not among them yet.) θ is what the
  trace holds; `constrain` maps it to the latents' values and `unconstrain`
  back. Without the declaration, a block's capabilities are in
  unconstrained coordinates already and every latent must be `:real`.

  The observations inside a block's target are invisible to the trace. Two
  optional capabilities show them: `:pointwise` (fn [theta inputs]) →
  {key log-lik}, each observation's log density, and `:simulate` (fn [theta
  inputs]) → {key value}, a fresh draw of each from its law. With them
  `diagnostics/pointwise-log-likelihood` (hence LOO and WAIC) and
  `infer/predictive` see the block's observations, at [site-address key]."
  (:require [org.replikativ.foerster.dist :as dist]))

(def ^:private required-capabilities #{:log-density :value+grad})

(defn- size [shape] (reduce * 1 shape))

(defrecord Block [description capabilities offsets dimension transforms])

;; =============================================================================
;; Transforms: θ (unconstrained) → x (the latent), per latent, with
;; log |det dx/dθ| and the pull-back of a gradient in x to one in θ.
;; =============================================================================

(defn- sigmoid [t] (/ 1.0 (+ 1.0 (Math/exp (- t)))))

(defn- softplus
  "log(1 + eᵗ), without overflow or underflow."
  [t]
  (if (> t 0.0) (+ t (Math/log1p (Math/exp (- t)))) (Math/log1p (Math/exp t))))

(defn- scalar-kind
  "[kind a w] of a scalar support: its transform and its bounds' constants
  (`a` the bound, `w` an interval's width), or nil."
  [support]
  (cond
    (= :real support) [:real 0.0 0.0]
    (= :positive support) [:positive 0.0 0.0]
    (and (vector? support) (= :lower (first support))) [:lower (double (second support)) 0.0]
    (and (vector? support) (= :upper (first support))) [:upper (double (second support)) 0.0]
    (and (vector? support) (= :interval (first support)))
    (let [[_ a b] support] [:interval (double a) (- (double b) (double a))])))

;; θ → x, dx/dθ, log|dx/dθ| and its derivative for each scalar kind, on
;; primitive doubles: a sampler calls these at every gradient

(defn- scalar-x ^double [kind ^double a ^double w ^double t]
  (case kind
    :real t
    :positive (Math/exp t)
    :lower (+ a (Math/exp t))
    :upper (- a (Math/exp t))
    :interval (+ a (* w (double (sigmoid t))))))

(defn- scalar-dx ^double [kind ^double _a ^double w ^double t]
  (case kind
    :real 1.0
    (:positive :lower) (Math/exp t)
    :upper (- (Math/exp t))
    :interval (let [s (double (sigmoid t))] (* w s (- 1.0 s)))))

(defn- scalar-lj ^double [kind ^double _a ^double w ^double t]
  (case kind
    :real 0.0
    (:positive :lower :upper) t
    ;; log σ(t) + log(1 − σ(t)) = −softplus(−t) − softplus(t): exact far
    ;; into the tails, where σ(t) rounds to 0 or 1
    :interval (- (Math/log w) (double (softplus t)) (double (softplus (- t))))))

(defn- scalar-dlj ^double [kind ^double _a ^double _w ^double t]
  (case kind
    :real 0.0
    (:positive :lower :upper) 1.0
    :interval (- 1.0 (* 2.0 (double (sigmoid t))))))

(defn- scalar-theta ^double [kind ^double a ^double w ^double x]
  (case kind
    :real x
    :positive (Math/log x)
    :lower (Math/log (- x a))
    :upper (Math/log (- a x))
    :interval (let [u (/ (- x a) w)] (Math/log (/ u (- 1.0 u))))))

(defn- support-error
  "Why `support` cannot be a latent of `shape`, or nil when it can."
  [support shape]
  (let [real-number? #(and (number? %) (not (NaN? (double %))) (not (infinite? (double %))))]
    (cond
      (#{:real :positive} support) nil
      (and (vector? support) (#{:lower :upper} (first support)))
      (when-not (and (= 2 (count support)) (real-number? (second support)))
        (str (first support) " takes one finite bound: " (pr-str [(first support) 0.0])))
      (and (vector? support) (= :interval (first support)))
      (when-not (and (= 3 (count support)) (every? real-number? (rest support))
                     (< (second support) (nth support 2)))
        "an :interval takes two finite bounds a < b: [:interval a b]")
      (#{:ordered :simplex} support)
      (cond (not= 1 (count shape)) (str support " is a vector latent: its :shape is [K]")
            (and (= :simplex support) (< (first shape) 2)) "a :simplex has at least two coordinates"
            (< (first shape) 1) (str support " needs at least one coordinate"))
      :else (str "unknown support " (pr-str support)))))

(defn- elementwise
  "A latent of `n` coordinates under the scalar transform of `support`."
  [support n]
  (let [[kind a w] (scalar-kind support)
        a (double a) w (double w) n (long n)]
    {:n-theta n :n-x n
     :constrain! (fn [^doubles th ti ^doubles xs xi]
                   (let [ti (long ti) xi (long xi)]
                     (dotimes [i n] (aset xs (+ xi i) (scalar-x kind a w (aget th (+ ti i)))))))
     :log-jacobian (fn [^doubles th ti]
                     (let [ti (long ti)]
                       (loop [i 0 acc 0.0]
                         (if (= i n) acc (recur (inc i) (+ acc (scalar-lj kind a w (aget th (+ ti i)))))))))
     :pullback! (fn [^doubles th ti ^doubles gx xi ^doubles out oi]
                  (let [ti (long ti) xi (long xi) oi (long oi)]
                    (dotimes [i n]
                      (let [t (aget th (+ ti i))]
                        (aset out (+ oi i) (+ (* (aget gx (+ xi i)) (scalar-dx kind a w t))
                                              (scalar-dlj kind a w t)))))))
     :unconstrain (fn [xs] (mapv #(scalar-theta kind a w (double %)) xs))}))

(defn- ordered
  "An increasing vector of `n`: x₀ = θ₀, xₖ = xₖ₋₁ + exp θₖ (Stan's ordered
  transform); log|J| = Σₖ₌₁ θₖ."
  [n]
  {:n-theta n :n-x n
   :constrain! (fn [^doubles th ti ^doubles xs xi]
                 (loop [k 0 prev 0.0]
                   (when (< k n)
                     (let [v (if (zero? k) (aget th ti) (+ prev (Math/exp (aget th (+ ti k)))))]
                       (aset xs (+ xi k) v)
                       (recur (inc k) v)))))
   :log-jacobian (fn [^doubles th ti]
                   (loop [k 1 acc 0.0] (if (>= k n) acc (recur (inc k) (+ acc (aget th (+ ti k)))))))
   ;; ∂xₖ/∂θⱼ is 1 for j = 0 and exp θⱼ for 0 < j ≤ k: the gradient is the
   ;; suffix sum of ∂f/∂x times that, plus ∂log|J|/∂θⱼ (1 for j > 0)
   :pullback! (fn [^doubles th ti ^doubles gx xi ^doubles out oi]
                (loop [k (dec n) suffix 0.0]
                  (when (>= k 0)
                    (let [suffix (+ suffix (aget gx (+ xi k)))]
                      (aset out (+ oi k) (if (zero? k) suffix (+ 1.0 (* suffix (Math/exp (aget th (+ ti k)))))))
                      (recur (dec k) suffix)))))
   :unconstrain (fn [xs] (let [xs (vec xs)]
                           (into [(double (first xs))]
                                 (map (fn [a b] (Math/log (- (double b) (double a)))) xs (rest xs)))))})

(defn- simplex
  "A probability vector of `k` by stick-breaking from k − 1 coordinates, as
  Stan's simplex_constrain: zⱼ = σ(θⱼ − log(k − 1 − j)), xⱼ = sⱼ·zⱼ of the
  stick sⱼ left, the last coordinate the stick that remains;
  log|J| = Σⱼ log sⱼ + log zⱼ + log(1 − zⱼ)."
  [k]
  (let [m (dec k)
        offsets (double-array (map #(Math/log (- m %)) (range m)))]
    {:n-theta m :n-x k
     :constrain! (fn [^doubles th ti ^doubles xs xi]
                   (loop [j 0 stick 1.0]
                     (if (= j m)
                       (aset xs (+ xi m) stick)
                       (let [x (* stick (sigmoid (- (aget th (+ ti j)) (aget offsets j))))]
                         (aset xs (+ xi j) x)
                         (recur (inc j) (- stick x))))))
     :log-jacobian (fn [^doubles th ti]
                     (loop [j 0 stick 1.0 acc 0.0]
                       (if (= j m)
                         acc
                         (let [a (- (aget th (+ ti j)) (aget offsets j))]
                           (recur (inc j) (* stick (sigmoid (- a)))
                                  (+ acc (Math/log stick) (- (softplus (- a))) (- (softplus a))))))))
     ;; reverse mode through the sticks: sⱼ₊₁ = sⱼ(1 − zⱼ), xⱼ = sⱼzⱼ
     :pullback! (fn [^doubles th ti ^doubles gx xi ^doubles out oi]
                  (let [sticks (double-array m) zs (double-array m)]
                    (loop [j 0 stick 1.0]
                      (when (< j m)
                        (let [z (double (sigmoid (- (aget th (+ ti j)) (aget offsets j))))]
                          (aset sticks j stick)
                          (aset zs j z)
                          (recur (inc j) (* stick (- 1.0 z))))))
                    (loop [j (dec m) s-bar (aget gx (+ xi m))]
                      (when (>= j 0)
                        (let [s (aget sticks j) z (aget zs j) g (aget gx (+ xi j))]
                          (aset out (+ oi j) (+ (* (- g s-bar) s z (- 1.0 z)) (- 1.0 (* 2.0 z))))
                          (recur (dec j) (+ (* g z) (* s-bar (- 1.0 z)) (/ 1.0 s))))))))
     :unconstrain (fn [xs]
                    (let [xs (vec xs)]
                      (loop [j 0 stick 1.0 out []]
                        (if (= j m)
                          out
                          (let [x (double (nth xs j)) z (/ x stick)]
                            (recur (inc j) (- stick x)
                                   (conj out (+ (Math/log (/ z (- 1.0 z))) (aget offsets j)))))))))}))

(defn- transform-of [{:keys [shape support] :or {support :real}}]
  (case support
    :ordered (ordered (first shape))
    :simplex (simplex (first shape))
    (elementwise support (size shape))))

(defn- transforms-of
  "Each latent's transform with its offsets in θ and in x."
  [latents]
  (loop [[l & more] latents ti (long 0) xi (long 0) out []]
    (if-not l
      out
      (let [tr (transform-of l)]
        (recur more (+ ti (long (:n-theta tr))) (+ xi (long (:n-x tr))) (conj out (assoc tr :ti ti :xi xi)))))))

(defn- constrain-array
  ^doubles [transforms n-x ^doubles th]
  (let [xs (double-array n-x)]
    (doseq [{:keys [constrain! ti xi]} transforms] (constrain! th ti xs xi))
    xs))

(defn- log-jacobian-of ^double [transforms ^doubles th]
  (reduce (fn [acc {:keys [log-jacobian ti]}] (+ acc (double (log-jacobian th ti)))) 0.0 transforms))

(defn- pullback
  "∇θ of f(x(θ)) + log|J(θ)| from `gx`, ∇ₓf at x(θ)."
  ^doubles [transforms n-theta ^doubles th ^doubles gx]
  (let [out (double-array n-theta)]
    (doseq [{:keys [pullback! ti xi]} transforms] (pullback! th ti gx xi out ti))
    out))

(defn- transformed
  "Capabilities over θ from capabilities over the constrained latents."
  [{:keys [log-density value+grad sample sample-log-density pointwise simulate] :as caps} transforms]
  (let [n-theta (reduce + 0 (map :n-theta transforms))
        n-x (reduce + 0 (map :n-x transforms))
        xs #(constrain-array transforms n-x %)
        log-j #(log-jacobian-of transforms %)
        unconstrain (fn [x] (let [x (vec x)]
                              (into [] (mapcat (fn [{:keys [unconstrain xi n-x]}] (unconstrain (subvec x xi (+ xi n-x)))))
                                    transforms)))]
    (cond-> (assoc caps
                   :log-density (fn [th inputs] (+ (double (log-density (xs th) inputs)) (log-j th)))
                   :value+grad (fn [^doubles th inputs]
                                 (let [[lp g] (value+grad (xs th) inputs)]
                                   ;; a capability may return its gradient as any sequence
                                   [(+ (double lp) (log-j th)) (pullback transforms n-theta th (double-array g))])))
      sample (assoc :sample (fn [inputs] (unconstrain (sample inputs))))
      pointwise (assoc :pointwise (fn [th inputs] (pointwise (xs th) inputs)))
      simulate (assoc :simulate (fn [th inputs] (simulate (xs th) inputs)))
      sample-log-density (assoc :sample-log-density
                                (fn [th inputs] (+ (double (sample-log-density (xs th) inputs)) (log-j th)))))))

(defn constrain
  "The latents' values at θ: θ mapped back through each latent's transform
  (identity for `:real`), as a vector in the latents' declared order (a
  `:simplex` of K takes K − 1 coordinates of θ and gives K)."
  [b theta]
  (let [transforms (:transforms b)]
    (vec (constrain-array transforms (reduce + 0 (map :n-x transforms)) (double-array theta)))))

(defn unconstrain
  "θ of the latents' values `x` (a vector in their declared order): the
  inverse of `constrain`."
  [b x]
  (let [x (vec x)]
    (into [] (mapcat (fn [{:keys [unconstrain xi n-x]}] (unconstrain (subvec x xi (+ xi n-x)))))
          (:transforms b))))

(defn jacobian
  "[log |det dx/dθ|, its gradient in θ] of block `b`'s transforms at θ: the
  term a constrained block adds to its density (0 for an unconstrained
  block). Subtract it for the mode in the latents' own coordinates."
  [b theta]
  (if (= :constrained (:block/coordinates (:description b)))
    (let [transforms (:transforms b)
          th (double-array theta)
          gx (double-array (reduce + 0 (map :n-x transforms)))]
      [(log-jacobian-of transforms th) (vec (pullback transforms (count theta) th gx))])
    [0.0 (vec (repeat (count theta) 0.0))]))

(def ^:private supports-doc
  ":real, :positive, [:lower a], [:upper b], [:interval a b], :ordered, :simplex")

(defn block
  "A block from its `description` and `capabilities` (see the namespace)."
  [description capabilities]
  (let [latents (:block/latents description)
        missing (remove (set (keys capabilities)) required-capabilities)]
    (when (seq missing)
      (throw (ex-info "A block lacks required capabilities"
                      {:type ::missing-capabilities :block (:block/id description)
                       :missing (vec missing)})))
    (doseq [{:keys [name shape support] :or {support :real}} latents]
      (when-let [why (support-error support shape)]
        (throw (ex-info (str "Latent " (pr-str name) " of block " (pr-str (:block/id description))
                             ": " why "; the supports are " supports-doc)
                        {:type ::unknown-support :block (:block/id description) :latent name
                         :support support :supports supports-doc}))))
    (when-let [constrained (and (not= :constrained (:block/coordinates description))
                                (seq (remove #(= :real (:support % :real)) latents)))]
      (throw (ex-info "Constrained latents need :block/coordinates :constrained"
                      {:type ::unsupported-support :block (:block/id description)
                       :latents (mapv :name constrained)})))
    (let [transforms (transforms-of latents)
          offsets (zipmap (map :name latents) (map :xi transforms))
          capabilities (if (= :constrained (:block/coordinates description))
                         (transformed capabilities transforms)
                         capabilities)]
      (->Block description capabilities offsets (reduce + 0 (map :n-theta transforms)) transforms))))

(defn dimension "The length of θ." [b] (:dimension b))

(defn capability
  "The capability `k` of block `b`, or nil."
  [b k]
  (get (:capabilities b) k))

(defn latent
  "The latent `name` of the latents' values `x` (`constrain`ed θ; θ itself
  for an unconstrained block): a double for a scalar, a vector otherwise."
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
      (let [known (::known (meta theta))]
        (if (and known (identical? block (:block known)) (= inputs (:inputs known))
                 (= theta (:theta known)))
          (:log-density known)
          ((capability block :log-density) (theta-array theta) inputs)))
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

(defn with-log-density
  "θ (a vector) carrying `lp`, the log target of `dist` (a `block-dist`) at
  θ, which a sampler has just computed with the gradient: replaying the site
  at the same block, inputs and θ scores it by `lp` instead of evaluating the
  density again. Any other block, inputs or θ (a changed copy of the vector
  keeps the metadata) evaluates the density as usual."
  [dist theta lp]
  (vary-meta theta assoc ::known {:block (:block dist) :inputs (:inputs dist)
                                  :theta theta :log-density (double lp)}))

(defn value+grad
  "[log target, gradient] of `dist` (a `block-dist`) at θ, both from the
  block's `:value+grad`; the gradient as a vector."
  [dist theta]
  (let [[lp g] ((capability (:block dist) :value+grad) (theta-array theta) (:inputs dist))]
    [(double lp) (vec g)]))

(defn value+grad!
  "As `value+grad` on a double array θ, the gradient a double array: for
  samplers that keep their state in arrays. θ is not modified."
  [dist ^doubles theta]
  (let [[lp g] ((capability (:block dist) :value+grad) theta (:inputs dist))]
    [(double lp) (double-array g)]))

(defn with-numeric-gradient
  "`capabilities` with a `:value+grad` computed from `:log-density` by
  central differences of step `h` (default 1e-6 times the coordinate's
  scale): 2n evaluations per gradient, accurate to about 1e-8 relative on a
  smooth density. An explicit opt-in for small blocks written in plain
  Clojure — foerster never differentiates numerically behind your back; for
  a compiled gradient, write the block with foerster-raster."
  ([capabilities] (with-numeric-gradient capabilities nil))
  ([{:keys [log-density] :as capabilities} {:keys [h] :or {h 1e-6}}]
   (assoc capabilities
          :value+grad
          (fn [^doubles th inputs]
            (let [n (alength th)
                  v (double (log-density th inputs))
                  g (double-array n)]
              (dotimes [i n]
                (let [x (aget th i)
                      step (* h (max 1.0 (Math/abs x)))
                      up (aclone th) down (aclone th)]
                  (aset up i (+ x step))
                  (aset down i (- x step))
                  (aset g i (/ (- (double (log-density up inputs)) (double (log-density down inputs)))
                               (* 2.0 step)))))
              [v g])))))
