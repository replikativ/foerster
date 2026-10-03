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
            [org.replikativ.foerster.dist :as dist]))

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
  [f theta0 {:keys [max-iterations tolerance history] :or {max-iterations 1000 tolerance 1e-8 history 10}}]
  (loop [theta (vec theta0) [fx g] (f theta0) pairs [] it 0]
    (if (or (< (norm g) (* tolerance (max 1.0 (norm theta)))) (= it max-iterations))
      {:theta theta :f fx :grad g :iterations it :converged? (< it max-iterations)}
      (let [p (let [p (direction g pairs)] (if (neg? (dot p g)) p (mapv - g)))
            slope (dot p g)
            [step theta' [fx' g']]
            (loop [a 1.0]
              (let [theta' (axpy a p theta) [fx' g' :as r] (f theta')]
                (if (or (and (not (NaN? fx')) (not (infinite? fx')) (<= fx' (+ fx (* 1e-4 a slope)))) (< a 1e-20))
                  [a theta' r]
                  (recur (* 0.5 a)))))]
        (if (< step 1e-20)
          {:theta theta :f fx :grad g :iterations it :converged? false}
          (let [s (mapv - theta' theta) y (mapv - g' g) sy (dot s y)
                pairs (if (> sy 1e-12)
                        (let [pairs (conj pairs [s y (/ 1.0 sy)])]
                          (if (> (count pairs) history) (subvec pairs 1) pairs))
                        pairs)]
            (recur theta' [fx' g'] pairs (inc it))))))))

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
