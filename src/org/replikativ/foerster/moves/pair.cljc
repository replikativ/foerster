(ns org.replikativ.foerster.moves.pair
  "Moves with auxiliary randomness, in one shape: a forward kernel that
  draws an auxiliary value and maps the state, and a backward kernel that
  scores the auxiliary value the reverse move would draw.

    forward   (fn [choices trace]) -> {:updates x' :log-q lq :reverse u'
                                       :log-jacobian lj}
              x' new values for some addresses (earlier ones included), lq
              the log density of the forward kernel's own draws, u' the
              auxiliary value the backward kernel would draw to go back, lj
              the log |det| of the deterministic map (0.0 for a discrete or
              volume-preserving move; required)
    backward  (fn [choices' u']) -> log L(u' ; x')

  `choices` is `foerster.runtime.trace/choices` ({address value}). The
  forward kernel draws from `foerster.random` (it runs in the particle's
  stream). A pair is SMCP3's move-reweight step (`foerster.inference.smc`,
  `:smcp3`) and a population transport (`foerster.program/transport-kernel`);
  an involutive MH move (`foerster.moves.involutive`) is the case whose map
  is an involution and whose backward kernel is the forward one, and
  `from-involution` makes it a pair. Valid laws and the Jacobian are the
  caller's obligation; they are not inferred.")

(defrecord Pair [forward backward])

(defn pair
  "The move of `forward` and `backward` (see the namespace)."
  [forward backward]
  (when-not (and (ifn? forward) (ifn? backward))
    (throw (ex-info "A pair is a forward kernel and a backward kernel, both functions"
                    {:type ::invalid-pair})))
  (->Pair forward backward))

(defn pair?
  "Whether `x` is a pair, or a map with `:forward` and `:backward`
  functions (the shape SMCP3 and transport take)."
  [x]
  (or (instance? Pair x) (and (map? x) (ifn? (:forward x)) (ifn? (:backward x)))))

(defn check!
  "`x` if it is a pair, else an error naming `what` takes."
  [x what]
  (when-not (pair? x)
    (throw (ex-info (str what " takes a pair: {:forward (fn [choices trace]) :backward (fn [choices' u'])}"
                         " (foerster.moves.pair)")
                    {:type ::invalid-pair})))
  x)

(defn from-involution
  "The pair of an involutive move {:propose (fn [choices]) -> {:aux u :log-q},
  :log-q (fn [choices u]), :involution (fn [choices u]) -> {:choices x' :aux
  u' :log-jacobian l}}: forward draws u and applies the involution, backward
  is q at the new state. So one move serves as an MCMC move
  (`k/involutive-kernel`) and as an SMCP3 step. A reversible jump (an
  involution that names `:removed` sites) stays an MCMC move only."
  [{:keys [propose log-q involution]}]
  (when-not (and (ifn? propose) (ifn? log-q) (ifn? involution))
    (throw (ex-info "An involutive move is {:propose :log-q :involution}" {:type ::invalid-pair})))
  (pair (fn [choices _trace]
          (let [{u :aux lq :log-q} (propose choices)
                {x' :choices u' :aux lj :log-jacobian removed :removed} (involution choices u)]
            (when (seq removed)
              (throw (ex-info "A reversible-jump involution (:removed) is an MCMC move only"
                              {:type ::removed-sites :removed removed})))
            {:updates x' :log-q lq :reverse u' :log-jacobian lj}))
        (fn [choices' u'] (log-q choices' u'))))
