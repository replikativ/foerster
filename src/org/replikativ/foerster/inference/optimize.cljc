(ns org.replikativ.foerster.inference.optimize
  "Point estimates and the Laplace approximation of a block's target.

    (optimize/map-estimate (block/block-dist b inputs) {:init θ0})
    ;; => {:theta θ* :latents {:mu … :sigma …} :log-density lp
    ;;     :iterations 23 :grad-norm 3e-9 :converged? true}

    (optimize/laplace (block/block-dist b inputs) {:init θ0})
    ;; => {:theta θ* :covariance Σ :log-evidence … :draw (fn [] {:mu … …})}

  `map-estimate` maximizes the log target by L-BFGS (Nocedal & Wright 2006,
  Alg. 7.4, with a backtracking Armijo line search). For a block with
  constrained latents it returns the mode in the latents' own coordinates
  (σ, p): the transforms' log-Jacobian is taken out of the target, as Stan's
  `optimize` does by default; `:jacobian? true` keeps it, giving the mode of
  the unconstrained θ.

  `laplace` approximates the posterior by a Gaussian in θ at the mode of
  the unconstrained target (Jacobian included, as Stan's `laplace`): mean
  θ*, covariance the inverse of the negative Hessian, which it takes by
  central differences of the block's gradient. Draws are mapped back
  through the transforms, so a positive latent stays positive. Its
  `:log-evidence` is log p(θ*) + d/2 log 2π − ½ log det(−H), exact for a
  Gaussian target. It is cheap and good when the posterior is close to
  Gaussian in θ; compare it with NUTS when that is in doubt."
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.diagnostics :as diagnostics]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]))

(defn- dot ^double [a b] (reduce + 0.0 (map * a b)))
(defn- axpy "a·x + y" [a x y] (mapv #(+ (* a %1) %2) x y))
(defn- norm [x] (Math/sqrt (dot x x)))

(defn- objective
  "f(θ) = −log target and its gradient, the Jacobian taken out unless
  `jacobian?`."
  [d jacobian?]
  (fn [theta]
    (let [[lp g] (block/value+grad d theta)]
      (if jacobian?
        [(- lp) (mapv - g)]
        (let [[lj gj] (block/jacobian (:block d) theta)]
          [(- lj lp) (mapv - gj g)])))))

(defn- direction
  "L-BFGS two-loop recursion: −H·g from the stored pairs (newest last)."
  [g pairs]
  (if (empty? pairs)
    (mapv - g)
    (let [[q alphas] (reduce (fn [[q alphas] [s y rho]]
                               (let [a (* rho (dot s q))]
                                 [(axpy (- a) y q) (conj alphas a)]))
                             [g []] (rseq pairs))
          [s y] (peek pairs)
          r (mapv #(* (/ (dot s y) (dot y y)) %) q)
          r (reduce (fn [r [[s y rho] a]]
                      (axpy (- a (* rho (dot y r))) s r))
                    r (map vector pairs (rseq alphas)))]
      (mapv - r))))

(defn- next-alpha
  "The diagonal estimate of the inverse Hessian after the curvature pair
  (s, y), sᵀy > 0: Zhang et al. (2022) Alg. 3, as BlackJAX's
  `lbfgs_recover_alpha`. A coordinate whose update is not positive and
  finite keeps its value."
  [alpha s y]
  (let [a (reduce + 0.0 (map (fn [al z] (* al z z)) alpha y))
        b (dot y s)
        c (reduce + 0.0 (map (fn [si al] (/ (* si si) al)) s alpha))]
    (mapv (fn [al si zi]
            (let [v (/ 1.0 (+ (/ a (* b al)) (/ (* zi zi) b) (- (/ (* a si si) (* b c al al)))))]
              (if (and (pos? v) (< v ##Inf)) v al)))
          alpha s y)))

(defn- lbfgs
  "Minimize `f` (θ → [f(θ) ∇f(θ)]) from θ0. With `record?`, `:path` holds
  every iterate as {:theta :grad :pairs :alpha}: the curvature pairs [s y
  1/sᵀy] L-BFGS held there (newest last), and the diagonal inverse Hessian
  estimate of Zhang et al.'s Alg. 3 over every pair so far. With
  `rel-obj-tolerance` it also stops once a step changes f by less than
  that, relative to |f| (Stan's tol_rel_obj)."
  [f theta0 {:keys [max-iterations tolerance history record? rel-obj-tolerance]
             :or {max-iterations 1000 tolerance 1e-8 history 10}}]
  (loop [theta (vec theta0) [fx g] (f theta0) pairs [] it 0
         alpha (when record? (vec (repeat (count theta0) 1.0)))
         path (when record? [{:theta (vec theta0) :grad g :pairs [] :alpha alpha}])]
    (if (or (< (norm g) (* tolerance (max 1.0 (norm theta)))) (= it max-iterations))
      {:theta theta :f fx :grad g :iterations it :converged? (< it max-iterations) :path path}
      (let [p (let [p (direction g pairs)] (if (neg? (dot p g)) p (mapv - g)))
            slope (dot p g)
            [step theta' [fx' g']]
            (loop [a 1.0]
              (let [theta' (axpy a p theta) [fx' _ :as r] (f theta')]
                (if (or (and (not (NaN? fx')) (not (infinite? fx')) (<= fx' (+ fx (* 1e-4 a slope)))) (< a 1e-20))
                  [a theta' r]
                  (recur (* 0.5 a)))))]
        (if (< step 1e-20)
          {:theta theta :f fx :grad g :iterations it :converged? false :path path}
          (let [s (mapv - theta' theta) y (mapv - g' g) sy (dot s y)
                pair? (> sy 1e-12)
                pairs (if pair?
                        (let [pairs (conj pairs [s y (/ 1.0 sy)])]
                          (if (> (count pairs) history) (subvec pairs 1) pairs))
                        pairs)
                alpha (if (and record? pair?) (next-alpha alpha s y) alpha)
                at {:theta theta' :grad g' :pairs pairs :alpha alpha}]
            (if (and rel-obj-tolerance
                     (< (/ (Math/abs (double (- fx fx'))) (max (Math/abs (double fx)) (Math/abs (double fx')) 1.0))
                        rel-obj-tolerance))
              {:theta theta' :f fx' :grad g' :iterations (inc it) :converged? true
               :path (when record? (conj path at))}
              (recur theta' [fx' g'] pairs (inc it) alpha
                     (when record? (conj path at))))))))))

(defn- latents-of [b x]
  (into {} (map (fn [{:keys [name]}] [name (block/latent b x name)])) (:block/latents (:description b))))

(defn map-estimate
  "The mode of the block site `d` (a `block/block-dist`), see the namespace.
  Options: `:init` θ0 (required unless the block can `:sample`),
  `:jacobian?` (false), `:max-iterations` (1000), `:tolerance` (1e-8, on
  the gradient norm relative to |θ|), `:history` (10). `:log-density` is
  the maximized target at the mode: in the latents' own coordinates (no
  Jacobian) unless `:jacobian?`."
  [d & [{:keys [init jacobian?] :as opts}]]
  (let [b (:block d)
        theta0 (or init (dist/draw d))
        {:keys [theta f grad iterations converged?]} (lbfgs (objective d jacobian?) theta0 opts)]
    {:theta theta
     :latents (latents-of b (block/constrain b theta))
     ;; f = −(the objective's log density) at θ, Jacobian taken out unless
     ;; jacobian?
     :log-density (- (double f))
     :iterations iterations
     :grad-norm (norm grad)
     :converged? converged?}))

(defn- hessian
  "−H of the log target at θ by central differences of the gradient."
  [d theta h]
  (let [n (count theta)
        grad-at #(second (block/value+grad d %))
        cols (vec (for [j (range n)]
                    (let [step (* h (max 1.0 (Math/abs (double (nth theta j)))))
                          up (grad-at (update theta j + step))
                          down (grad-at (update theta j - step))]
                      (mapv #(/ (- %2 %1) (* 2.0 step)) up down))))]
    ;; cols[j][i] = −∂²/∂θi∂θj; symmetrize
    (vec (for [i (range n)] (vec (for [j (range n)] (* 0.5 (+ (get-in cols [j i]) (get-in cols [i j])))))))))

(defn- cholesky
  "Lower L with L·Lᵀ = a, or nil when a is not positive definite."
  [a]
  (let [n (count a)]
    (reduce (fn [l [i j]]
              (when l
                (let [s (- (get-in a [i j]) (reduce + 0.0 (map #(* (get-in l [i %]) (get-in l [j %])) (range j))))]
                  (if (= i j)
                    (when (pos? s) (assoc-in l [i i] (Math/sqrt s)))
                    (assoc-in l [i j] (/ s (get-in l [j j])))))))
            (vec (repeat n (vec (repeat n 0.0))))
            (for [i (range n) j (range (inc i))] [i j]))))

(defn- lower-inverse [l]
  (let [n (count l)]
    (reduce (fn [inv i]
              (reduce (fn [inv j]
                        (assoc-in inv [i j]
                                  (if (= i j)
                                    (/ 1.0 (get-in l [i i]))
                                    (/ (- (reduce + 0.0 (map #(* (get-in l [i %]) (get-in inv [% j])) (range j i))))
                                       (get-in l [i i])))))
                      inv (range (inc i))))
            (vec (repeat n (vec (repeat n 0.0))))
            (range n))))

(defn laplace
  "The Laplace approximation of the block site `d`, see the namespace.
  Options as `map-estimate`, and `:h` (1e-5), the relative step of the
  Hessian's differences. Returns {:theta :latents :covariance (of θ)
  :log-evidence :draw}; `(draw)` gives one approximate posterior draw of
  the latents, mapped back to their own coordinates. Throws
  `::not-positive-definite` when the mode is not a strict maximum."
  [d & [{:keys [h] :or {h 1e-5} :as opts}]]
  (let [b (:block d)
        {:keys [theta log-density] :as mode} (map-estimate d (assoc opts :jacobian? true))
        neg-h (hessian d theta h)
        l (or (cholesky neg-h)
              (throw (ex-info "The Hessian at the mode is not negative definite"
                              {:type ::not-positive-definite :theta theta})))
        n (count theta)
        ;; Σ = (−H)⁻¹ = L⁻ᵀ L⁻¹; a draw is θ* + L⁻ᵀ z
        li (lower-inverse l)
        cov (vec (for [i (range n)] (vec (for [j (range n)] (reduce + 0.0 (map #(* (get-in li [% i]) (get-in li [% j])) (range n)))))))
        log-det (* 2.0 (reduce + 0.0 (map #(Math/log (get-in l [% %])) (range n))))
        std-normal (dist/normal 0.0 1.0)]
    (assoc (dissoc mode :latents)
           :latents (latents-of b (block/constrain b theta))
           :covariance cov
           :log-evidence (+ log-density (* 0.5 n (Math/log (* 2.0 Math/PI))) (* -0.5 log-det))
           :draw (fn []
                   (let [z (vec (repeatedly n #(dist/draw std-normal)))
                         x (mapv (fn [i] (+ (nth theta i) (reduce + 0.0 (map #(* (get-in li [% i]) (nth z %)) (range n)))))
                                 (range n))]
                     (latents-of b (block/constrain b x)))))))

;; =============================================================================
;; Pathfinder
;; =============================================================================

(def ^:private half-log-2pi 0.9189385332046727)

(defn- std-normals [n] (let [s (dist/normal 0.0 1.0)] (vec (repeatedly n #(dist/draw s)))))

(defn- transpose [a] (if (empty? a) [] (apply mapv vector a)))

(defn- mat* [a b]
  (let [bt (transpose b)] (mapv (fn [row] (mapv #(dot row %) bt)) a)))

(defn- upper-inverse
  "The inverse of the upper triangular `r` (nonzero diagonal)."
  [r]
  (let [n (count r)]
    (reduce (fn [inv j]
              ;; column j: back substitution of r·x = eⱼ
              (reduce (fn [inv i]
                        (assoc-in inv [i j]
                                  (/ (- (if (= i j) 1.0 0.0)
                                        (reduce + 0.0 (map #(* (get-in r [i %]) (get-in inv [% j])) (range (inc i) (inc j)))))
                                     (get-in r [i i]))))
                      inv (range j -1 -1)))
            (vec (repeat n (vec (repeat n 0.0))))
            (range n))))

(defn- inverse-hessian-factors
  "L-BFGS's inverse Hessian estimate from initial diagonal `alpha` and the
  curvature `pairs` [s y ρ] (newest last) as diagonal plus low rank,
  H = diag(α) + β·γ·βᵀ with β = [diag(α)·Y, S] (d × 2m) and
  γ = [[0, −R⁻¹], [−R⁻ᵀ, R⁻ᵀ(D + YᵀαY)R⁻¹]], R the upper triangle of SᵀY
  and D its diagonal: the compact form of the BFGS updates (Byrd, Nocedal
  & Schnabel 1994), as Zhang et al. (2022, II.2) use it. Returns
  {:alpha :beta (columns) :gamma}."
  [pairs alpha]
  (let [ss (mapv first pairs) ys (mapv second pairs) m (count pairs)
        sty (vec (for [i (range m)] (vec (for [j (range m)] (dot (ss i) (ys j))))))
        rinv (upper-inverse (vec (for [i (range m)] (vec (for [j (range m)] (if (<= i j) (get-in sty [i j]) 0.0))))))
        mid (vec (for [i (range m)]
                   (vec (for [j (range m)]
                          (+ (reduce + 0.0 (map (fn [al a b] (* al a b)) alpha (ys i) (ys j)))
                             (if (= i j) (get-in sty [i i]) 0.0))))))
        lower-right (mat* (transpose rinv) (mat* mid rinv))
        neg-rinv (mapv #(mapv - %) rinv)
        neg-rinv-t (transpose neg-rinv)]
    {:alpha alpha
     :beta (into (mapv #(mapv * alpha %) ys) ss)
     :gamma (into (mapv (fn [i] (into (vec (repeat m 0.0)) (neg-rinv i))) (range m))
                  (mapv (fn [i] (into (neg-rinv-t i) (lower-right i))) (range m)))}))

(defn- factors-times
  "H·v for H = diag(α) + β·γ·βᵀ."
  [{:keys [alpha beta gamma]} v]
  (let [bv (mapv #(dot % v) beta)
        gbv (mapv #(dot % bv) gamma)]
    (reduce (fn [acc [b c]] (axpy c b acc)) (mapv * alpha v) (map vector beta gbv))))

(defn- dense-inverse-hessian
  "diag(α) + β·γ·βᵀ as a dense matrix (rows): O(d²·m)."
  [{:keys [alpha beta gamma]}]
  (let [n (count alpha)
        bg (mapv (fn [row] (mapv (fn [col] (reduce + 0.0 (map * row col))) (transpose gamma)))
                 (transpose beta))]
    (vec (for [i (range n)]
           (vec (for [j (range n)]
                  (+ (if (= i j) (nth alpha i) 0.0)
                     (reduce + 0.0 (map (fn [x b] (* x (nth b j))) (bg i) beta)))))))))

(defn- thin-qr
  "B = Q·R for the columns `bs`: Q orthonormal columns, one for each column
  of B independent of those before it, by Gram–Schmidt twice (Giraud et
  al. 2005); R (k × n) as rows."
  [bs]
  (let [[qs rcols]
        (reduce (fn [[qs rcols] b]
                  (let [[v coef] (reduce (fn [[v coef] _]
                                           (let [cs (mapv #(dot % v) qs)]
                                             [(reduce (fn [v [q c]] (axpy (- c) q v)) v (map vector qs cs))
                                              (mapv + coef cs)]))
                                         [b (vec (repeat (count qs) 0.0))] [1 2])
                        nv (norm v)]
                    (if (> nv (* 1e-10 (max 1e-300 (norm b))))
                      [(conj qs (mapv #(/ % nv) v)) (conj rcols (conj coef nv))]
                      [qs (conj rcols coef)])))
                [[] []] bs)
        k (count qs)]
    [qs (vec (for [i (range k)] (mapv #(get % i 0.0) rcols)))]))

(defn- gaussian
  "{:mean :at :draw :factors :covariance} of N(mean, H), H = diag(α) +
  β·γ·βᵀ (`factors`), or nil when H is not positive definite, by Zhang et
  al.'s Alg. 4: with √α⁻¹β = Q·R (thin) and L·Lᵀ = I + R·γ·Rᵀ, a draw is
  mean + √α ⊙ (Q(L − I)Qᵀz + z), log det H = Σ log α + 2 Σ log Lᵢᵢ —
  O(d·m²) to set up and O(d·m) per draw. `(at z)` maps standard normals z
  to [θ log q(θ)], `(draw)` does so for fresh ones; `:covariance` is a
  delay of the dense H."
  [mean {:keys [alpha beta gamma] :as factors}]
  (let [n (count mean)
        sqrt-alpha (mapv #(Math/sqrt %) alpha)
        [qs r] (thin-qr (mapv #(mapv / % sqrt-alpha) beta))
        k (count qs)
        rgr (mat* r (mat* gamma (transpose r)))]
    (when-let [l (if (zero? k) [] (cholesky (vec (for [i (range k)] (update (rgr i) i + 1.0)))))]
      (let [log-det (+ (reduce + 0.0 (map #(Math/log %) alpha))
                       (* 2.0 (reduce + 0.0 (map #(Math/log (get-in l [% %])) (range k)))))
            at (fn [z]
                 (let [t (mapv #(dot % z) qs)
                       ;; (L − I)·t
                       w (mapv (fn [i] (- (reduce + 0.0 (map #(* (get-in l [i %]) (nth t %)) (range (inc i)))) (nth t i)))
                               (range k))
                       u (reduce (fn [acc [q c]] (axpy c q acc)) z (map vector qs w))]
                   [(mapv (fn [m s x] (+ m (* s x))) mean sqrt-alpha u)
                    (- (* -0.5 (dot z z)) (* 0.5 log-det) (* n half-log-2pi))]))]
        {:mean mean :factors factors :covariance (delay (dense-inverse-hessian factors))
         :at at :draw #(at (std-normals n))}))))

(defn- approximation
  "The Gaussian at an L-BFGS iterate: μ = θ − H·∇f (f = −log p), Σ = H,
  H from the iterate's `pairs` and initial diagonal `alpha`; nil when H is
  not positive definite."
  [theta grad pairs alpha]
  (let [factors (inverse-hessian-factors pairs alpha)]
    (gaussian (mapv - theta (factors-times factors grad)) factors)))

(defn- initial-diagonal
  "The initial inverse Hessian diagonal at an iterate: L-BFGS's own γ·I,
  γ = sᵀy/yᵀy of the newest pair (`:lbfgs`), or Zhang et al.'s Alg. 3
  estimate along the path (`:recovered`, as Stan and BlackJAX)."
  [diagonal pairs alpha]
  (case diagonal
    :lbfgs (let [[s y] (peek pairs)] (vec (repeat (count s) (/ (dot s y) (dot y y)))))
    :recovered alpha))

(defn- log-target
  "The log target of `d` at θ, the log density alone (no gradient): −∞ for
  a non-finite value."
  [d theta]
  (let [lp (double (dist/logpdf d theta))]
    (if (or (NaN? lp) (infinite? lp)) ##-Inf lp)))

(defn- single-path
  "One Pathfinder path from θ0: the Gaussian along the L-BFGS path with the
  largest ELBO estimate. Every iterate's estimate uses the same standard
  normals (common random numbers), so their differences are not swamped by
  Monte Carlo noise."
  [d theta0 {:keys [elbo-draws history diagonal] :as opts}]
  (let [n (count theta0)
        zs (vec (repeatedly elbo-draws #(std-normals n)))
        {:keys [path iterations]} (lbfgs (objective d true) theta0
                                         (merge {:rel-obj-tolerance (* 1e4 2.220446049250313e-16)}
                                                opts {:record? true :history history}))
        candidates (keep (fn [{:keys [theta grad pairs alpha]}]
                           (when (seq pairs)
                             (when-let [q (approximation theta grad pairs (initial-diagonal diagonal pairs alpha))]
                               (let [lw (mapv #(let [[x lq] ((:at q) %)] (- (log-target d x) lq)) zs)]
                                 (assoc q :elbo (/ (reduce + lw) elbo-draws))))))
                         path)]
    (when (seq candidates)
      (assoc (apply max-key :elbo candidates) :iterations iterations))))

(defn pathfinder
  "Pathfinder (Zhang, Carpenter, Gelman & Vehtari 2022) on the block site
  `d` (a `block/block-dist`): a Gaussian approximation of its target in the
  unconstrained θ, and approximate draws.

  Each of `:paths` runs L-BFGS on the target (Jacobian included, as
  `laplace`) from `:init` (default 0) plus a uniform jitter in (−2, 2) per
  coordinate. At every iterate L-BFGS's inverse Hessian estimate H and the
  gradient give a Gaussian N(θ + H·∇log p, H); a Monte Carlo estimate of
  the ELBO from `:elbo-draws` draws picks the best one along the path. The
  paths' Gaussians then give `:draws` draws in all, importance-resampled
  against the target with Pareto-smoothed weights.

  H is kept as a diagonal plus a rank-2m term (m ≤ `:history` pairs), the
  compact form of the BFGS updates, and a Gaussian is drawn from and
  scored by Zhang et al.'s Alg. 4: O(d·m²) per iterate and O(d·m) per
  draw, no d × d matrix (only the returned `:covariance` is dense). Its
  diagonal is Zhang et al.'s Alg. 3 estimate recovered along the path
  (`:diagonal :recovered`, the default, as Stan and BlackJAX), or L-BFGS's
  own γ·I (`:diagonal :lbfgs`: the inverse Hessian the optimizer used).

  Options: `:init`, `:paths` (4), `:draws` (1000), `:elbo-draws` (30),
  `:history` (6, L-BFGS pairs), `:diagonal` (`:recovered`), `:max-iterations`
  (1000), `:tolerance`,
  `:rel-obj-tolerance` (1e4·ε, as Stan: a step that changes the target by
  less, relatively, ends the path).
  Returns {:mean :covariance (the best path's Gaussian, by ELBO) :elbo
  :draws [θ …] :pareto-k :paths [{:elbo :iterations} …] :draw}; `(draw)`
  gives a draw from the best Gaussian, mapped back to the latents. A k̂
  above 0.7 says the Gaussians miss part of the target: use them to start a
  sampler, not as the posterior. Throws `::no-approximation` when no path
  yields a Gaussian or none of its draws has positive target density."
  [d & [{:keys [init paths draws elbo-draws history diagonal]
         :or {paths 4 draws 1000 elbo-draws 30 history 6 diagonal :recovered}
         :as opts}]]
  (doseq [[k v] {:paths paths :draws draws :elbo-draws elbo-draws :history history}]
    (when-not (pos-int? v)
      (throw (ex-info (str k " is a positive count") {:type ::invalid-option k v}))))
  (when-not (#{:lbfgs :recovered} diagonal)
    (throw (ex-info ":diagonal is :lbfgs or :recovered" {:type ::invalid-option :diagonal diagonal})))
  (let [b (:block d)
        n (or (some-> init count) (:dimension b))
        base (or init (vec (repeat n 0.0)))
        jitter (dist/uniform -2.0 2.0)
        per-path (max 1 (long (Math/ceil (/ (double draws) paths))))
        qs (vec (keep (fn [_]
                        (single-path d (mapv #(+ % (dist/draw jitter)) base)
                                     (assoc opts :elbo-draws elbo-draws :history history :diagonal diagonal)))
                      (range paths)))]
    (when (empty? qs)
      (throw (ex-info "Pathfinder found no Gaussian approximation" {:type ::no-approximation})))
    (let [proposals (vec (for [q qs _ (range per-path)]
                           (let [[x lq] ((:draw q))] [x (- (log-target d x) lq)])))
          _ (when (every? #(= ##-Inf (second %)) proposals)
              (throw (ex-info "Pathfinder's draws all have zero target density"
                              {:type ::no-approximation :proposals (count proposals)})))
          ;; one proposal has nothing to smooth
          [lw k] (if (= 1 (count proposals)) [[0.0] ##NaN] (diagnostics/psis (mapv second proposals) 1.0))
          picks (m/systematic-resample (mapv #(Math/exp %) lw) draws)
          best (apply max-key :elbo qs)]
      {:mean (:mean best)
       :covariance @(:covariance best)
       :elbo (:elbo best)
       :draws (mapv #(first (nth proposals %)) picks)
       :pareto-k k
       :paths (mapv #(select-keys % [:elbo :iterations]) qs)
       :draw (fn [] (latents-of b (block/constrain b (first ((:draw best))))))})))
