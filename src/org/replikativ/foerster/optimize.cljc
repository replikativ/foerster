(ns org.replikativ.foerster.optimize
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

(defn- lbfgs
  "Minimize `f` (θ → [f(θ) ∇f(θ)]) from θ0. With `record?`, `:path` holds
  every iterate as {:theta :grad :pairs}, the curvature pairs [s y 1/sᵀy]
  L-BFGS held there (newest last). With `rel-obj-tolerance` it also stops
  once a step changes f by less than that, relative to |f| (Stan's
  tol_rel_obj)."
  [f theta0 {:keys [max-iterations tolerance history record? rel-obj-tolerance]
             :or {max-iterations 1000 tolerance 1e-8 history 10}}]
  (loop [theta (vec theta0) [fx g] (f theta0) pairs [] it 0
         path (when record? [{:theta (vec theta0) :grad g :pairs []}])]
    (if (or (< (norm g) (* tolerance (max 1.0 (norm theta)))) (= it max-iterations))
      {:theta theta :f fx :grad g :iterations it :converged? (< it max-iterations) :path path}
      (let [p (let [p (direction g pairs)] (if (neg? (dot p g)) p (mapv - g)))
            slope (dot p g)
            [step theta' [fx' g']]
            (loop [a 1.0]
              (let [theta' (axpy a p theta) [fx' g' :as r] (f theta')]
                (if (or (and (not (NaN? fx')) (not (infinite? fx')) (<= fx' (+ fx (* 1e-4 a slope)))) (< a 1e-20))
                  [a theta' r]
                  (recur (* 0.5 a)))))]
        (if (< step 1e-20)
          {:theta theta :f fx :grad g :iterations it :converged? false :path path}
          (let [s (mapv - theta' theta) y (mapv - g' g) sy (dot s y)
                pairs (if (> sy 1e-12)
                        (let [pairs (conj pairs [s y (/ 1.0 sy)])]
                          (if (> (count pairs) history) (subvec pairs 1) pairs))
                        pairs)]
            (if (and rel-obj-tolerance
                     (< (/ (Math/abs (- fx fx')) (max (Math/abs fx) (Math/abs fx') 1.0)) rel-obj-tolerance))
              {:theta theta' :f fx' :grad g' :iterations (inc it) :converged? true
               :path (when record? (conj path {:theta theta' :grad g' :pairs pairs}))}
              (recur theta' [fx' g'] pairs (inc it)
                     (when record? (conj path {:theta theta' :grad g' :pairs pairs}))))))))))

(defn- latents-of [b x]
  (into {} (map (fn [{:keys [name]}] [name (block/latent b x name)])) (:block/latents (:description b))))

(defn map-estimate
  "The mode of the block site `d` (a `block/block-dist`), see the namespace.
  Options: `:init` θ0 (required unless the block can `:sample`),
  `:jacobian?` (false), `:max-iterations` (1000), `:tolerance` (1e-8, on
  the gradient norm relative to |θ|), `:history` (10)."
  [d & [{:keys [init jacobian?] :as opts}]]
  (let [b (:block d)
        theta0 (or init (dist/draw d))
        {:keys [theta f grad iterations converged?]} (lbfgs (objective d jacobian?) theta0 opts)]
    {:theta theta
     :latents (latents-of b (block/constrain b theta))
     :log-density (first (block/value+grad d theta))
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

(defn- inverse-hessian
  "The inverse Hessian estimate of L-BFGS from its `pairs` [s y ρ] (newest
  last), dense: the BFGS inverse updates applied in order to γ·I, γ = sᵀy/yᵀy
  of the newest pair."
  [pairs n]
  (let [[s y] (peek pairs)
        gamma (/ (dot s y) (dot y y))]
    (reduce (fn [h [s y rho]]
              ;; H⁺ = H − ρ(s·(Hy)ᵀ + (Hy)·sᵀ) + (ρ²·yᵀHy + ρ)·s·sᵀ
              (let [hy (mapv #(dot % y) h)
                    c (+ (* rho rho (dot y hy)) rho)]
                (vec (for [i (range n)]
                       (vec (for [j (range n)]
                              (+ (get-in h [i j])
                                 (* (- rho) (+ (* (nth s i) (nth hy j)) (* (nth hy i) (nth s j))))
                                 (* c (nth s i) (nth s j)))))))))
            (vec (for [i (range n)] (vec (for [j (range n)] (if (= i j) gamma 0.0)))))
            pairs)))

(def ^:private half-log-2pi 0.9189385332046727)

(defn- std-normals [n] (let [s (dist/normal 0.0 1.0)] (vec (repeatedly n #(dist/draw s)))))

(defn- gaussian
  "{:mean :covariance :at :draw} of N(mean, cov), or nil when cov is not
  positive definite: `(at z)` maps standard normals z to [θ log q(θ)],
  `(draw)` does so for fresh ones."
  [mean cov]
  (when-let [l (cholesky cov)]
    (let [n (count mean)
          log-det (* 2.0 (reduce + 0.0 (map #(Math/log (get-in l [% %])) (range n))))
          at (fn [z]
               [(mapv (fn [i] (+ (nth mean i) (reduce + 0.0 (map #(* (get-in l [i %]) (nth z %)) (range (inc i))))))
                      (range n))
                (- (* -0.5 (dot z z)) (* 0.5 log-det) (* n half-log-2pi))])]
      {:mean mean :covariance cov :at at :draw #(at (std-normals n))})))

(defn- log-target [d theta]
  (let [lp (first (block/value+grad d theta))]
    (if (or (NaN? lp) (infinite? lp)) ##-Inf lp)))

(defn- single-path
  "One Pathfinder path from θ0: the Gaussian along the L-BFGS path with the
  largest ELBO estimate. Every iterate's estimate uses the same standard
  normals (common random numbers), so their differences are not swamped by
  Monte Carlo noise."
  [d theta0 {:keys [elbo-draws history] :as opts}]
  (let [n (count theta0)
        zs (vec (repeatedly elbo-draws #(std-normals n)))
        {:keys [path iterations]} (lbfgs (objective d true) theta0
                                         (merge {:rel-obj-tolerance (* 1e4 2.220446049250313e-16)}
                                                opts {:record? true :history history}))
        candidates (keep (fn [{:keys [theta grad pairs]}]
                           (when (seq pairs)
                             (let [h (inverse-hessian pairs n)]
                               ;; μ = θ − H·∇f, f = −log p
                               (when-let [q (gaussian (mapv - theta (mapv #(dot % grad) h)) h)]
                                 (let [lw (mapv #(let [[x lq] ((:at q) %)] (- (log-target d x) lq)) zs)]
                                   (assoc q :elbo (/ (reduce + lw) elbo-draws)))))))
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

  Options: `:init`, `:paths` (4), `:draws` (1000), `:elbo-draws` (30),
  `:history` (6, L-BFGS pairs), `:max-iterations` (1000), `:tolerance`,
  `:rel-obj-tolerance` (1e4·ε, as Stan: a step that changes the target by
  less, relatively, ends the path).
  Returns {:mean :covariance (the best path's Gaussian, by ELBO) :elbo
  :draws [θ …] :pareto-k :paths [{:elbo :iterations} …] :draw}; `(draw)`
  gives a draw from the best Gaussian, mapped back to the latents. A k̂
  above 0.7 says the Gaussians miss part of the target: use them to start a
  sampler, not as the posterior. Throws `::no-approximation` when no path
  yields a Gaussian."
  [d & [{:keys [init paths draws elbo-draws history]
         :or {paths 4 draws 1000 elbo-draws 30 history 6}
         :as opts}]]
  (let [b (:block d)
        n (or (some-> init count) (:dimension b))
        base (or init (vec (repeat n 0.0)))
        jitter (dist/uniform -2.0 2.0)
        per-path (max 1 (long (Math/ceil (/ (double draws) paths))))
        qs (vec (keep (fn [_]
                        (single-path d (mapv #(+ % (dist/draw jitter)) base)
                                     (assoc opts :elbo-draws elbo-draws :history history)))
                      (range paths)))]
    (when (empty? qs)
      (throw (ex-info "Pathfinder found no Gaussian approximation" {:type ::no-approximation})))
    (let [proposals (vec (for [q qs _ (range per-path)]
                           (let [[x lq] ((:draw q))] [x (- (log-target d x) lq)])))
          [lw k] (diagnostics/psis (mapv second proposals) 1.0)
          picks (m/systematic-resample (mapv #(Math/exp %) lw) draws)
          best (apply max-key :elbo qs)]
      {:mean (:mean best)
       :covariance (:covariance best)
       :elbo (:elbo best)
       :draws (mapv #(first (nth proposals %)) picks)
       :pareto-k k
       :paths (mapv #(select-keys % [:elbo :iterations]) qs)
       :draw (fn [] (latents-of b (block/constrain b (first ((:draw best))))))})))
