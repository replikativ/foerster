(ns org.replikativ.foerster.gsdm
  "Small native amortized-abduction experiment, deliberately outside the jar.
  z, ux, uy ~ N(0,1); x=z+ux; y=2x+z+uy. Encode evidence as [x,y-2x].
  The learned target is z | x,y, analytically N((y-x)/3, 1/3).
  DDPM draws are approximate and are never advertised as scored proposals."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [raster.dl.gsdm :as g]
            [raster.dl.diffusion :as diffusion]
            [raster.params :as params]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn scm []
  (spin
   (let [z (sample (dist/normal 0 1) :id :z)
         x (sample (dist/normal z 1) :id :x)
         y (sample (dist/normal (+ (* 2 x) z) 1) :id :y)]
     {:z z :x x :y y})))

(defn oracle [{:keys [x y]}]
  (let [mu (/ (- y x) 3.0)]
    {:z-mean mu :z-variance (/ 1.0 3)
     :y-do-z-zero-mean (- y (* 3 mu)) :y-do-z-zero-variance 3.0
     :y-do-x-zero (- y (* 2 x))}))

(defn abduction
  "Translate one approximate z posterior draw into exact factual residuals.
  Zero log weight means equal-weight approximate draws, not exact inference."
  [{:keys [x y]} z]
  {:factual {:z z :x x :y y} :weight 0.0 :unsupported []
   :noise {:z z :x (- x z) :y (- y (* 2 x) z)}})

(defn prepare
  "Compile once per architecture. This is the expensive step; retain its
  result in a REPL. Training changes the shared arrays in flat/tree weights."
  ([] (prepare {}))
  ([{:keys [emb-dim n-layers timesteps dense? prediction]
     :or {emb-dim 8 n-layers 1 timesteps 64 dense? false prediction :x0}}]
   (let [started (System/nanoTime)
         config (assoc (g/make-gsdm-config :emb-dim emb-dim :n-layers n-layers
                                           :n-heads 2 :n-spaces 1 :timesteps timesteps)
                       :prediction prediction :dense? dense?)
         flat (g/init-gsdm-weights config) weights (g/flat->tree flat n-layers)
         loss-var (g/make-gsdm-loss config)
         training (params/compile-train-step loss-var)
         ;; The public loss body contains the same graph-transformer forward
         ;; expression. Export its prediction before the final loss reduction,
         ;; retaining the parameter spec, then compile with Raster's public API.
         [let-sym bindings] (g/gsdm-loss-body emb-dim 2 1)
         forward-body (list let-sym (vec (drop-last 2 bindings)) 'pred)
         forward-sym (symbol (str "experiment-forward-" n-layers "L-" emb-dim "d"))
         forward-var (binding [*ns* (the-ns 'org.replikativ.foerster.gsdm)]
                       (eval (list 'raster.core/deftm forward-sym
                                   ['w ':- (g/weights-spec n-layers)
                                    'values ':- '(Array double) 'spaces ':- '(Array long)
                                    'target ':- '(Array double) 'states ':- '(Array long)
                                    'pos-emb ':- '(Array double) 'src-edges ':- '(Array long)
                                    'dst-edges ':- '(Array long) 't ':- 'Double
                                    'n-vars ':- 'Long 'n-edges ':- 'Long]
                                   ':- '(Array double) forward-body))
                       (ns-resolve *ns* forward-sym))
         forward (params/compile-aot forward-var)
         edges (if dense? (g/fully-connected-edges 3)
                   {:src-edges (long-array [0 1 2 0 1 0 2])
                    :dst-edges (long-array [0 1 2 1 0 2 0]) :n-edges 7})
         spaces (long-array 3) states (long-array [g/LATENT g/OBSERVED g/OBSERVED])
         betas (double-array (for [i (range timesteps)]
                               (let [a (Math/pow (Math/cos (* (/ (+ (/ i (double timesteps)) 0.008) 1.008) (/ Math/PI 2))) 2)
                                     b (Math/pow (Math/cos (* (/ (+ (/ (inc i) (double timesteps)) 0.008) 1.008) (/ Math/PI 2))) 2)]
                                 (min 0.999 (max 0.0001 (- 1 (/ b a)))))))]
     {:config config :flat flat :weights weights :training training
      :adam ((:init-state training) weights) :iteration (atom 0)
      :forward forward :spaces spaces :states states :edges edges
      :positions (g/compute-position-embeddings spaces 3 emb-dim)
      :betas betas :alpha-bars (diffusion/compute-alphas-cumprod betas)
      :preparation-seconds (/ (- (System/nanoTime) started) 1e9)})))

(defn fresh
  "Fresh parameters/optimizer with the SAME compiled architecture. Useful for
  matched objective comparisons without another compiler invocation."
  [compiled prediction]
  (let [config (assoc (:config compiled) :prediction prediction)
        flat (g/init-gsdm-weights config)
        weights (g/flat->tree flat (:n-layers config))]
    (assoc compiled :config config :flat flat :weights weights
           :adam ((get-in compiled [:training :init-state]) weights)
           :iteration (atom 0) :preparation-seconds 0.0 :inference-only? false)))

(defn- predict [experiment values t]
  (let [{:keys [weights forward spaces states positions edges]} experiment]
    (forward weights values spaces values states positions
             (:src-edges edges) (:dst-edges edges) (double t) 3 (long (:n-edges edges)))))

(defn training-example [experiment ^java.util.Random rng]
  (let [z (.nextGaussian rng) a (+ z (.nextGaussian rng)) b (+ z (.nextGaussian rng))
        t (.nextInt rng (count (:betas experiment)))
        abar (aget ^doubles (:alpha-bars experiment) t)
        epsilon (.nextGaussian rng)
        zt (+ (* (Math/sqrt abar) z) (* (Math/sqrt (- 1 abar)) epsilon))
        target (case (get-in experiment [:config :prediction])
                 :v (- (* (Math/sqrt abar) epsilon) (* (Math/sqrt (- 1 abar)) z))
                 :x0 z)]
    {:values (double-array [zt a b]) :target (double-array [target a b]) :z z :t t}))

(defn- predict-x0 [experiment values t]
  (let [raw (aget ^doubles (predict experiment values t) 0)
        abar (aget ^doubles (:alpha-bars experiment) t)]
    (case (get-in experiment [:config :prediction])
      :v (- (* (Math/sqrt abar) (aget ^doubles values 0)) (* (Math/sqrt (- 1 abar)) raw))
      :x0 raw)))

(defn train!
  "Fresh simulator draws at every update. Continue the optimizer in the REPL;
  use a distinct seed for each chunk. Returns a timing/loss receipt."
  [experiment updates {:keys [seed rate] :or {seed 101 rate 0.001}}]
  (when (:inference-only? experiment)
    (throw (ex-info "This checkpoint contains inference weights, not optimizer state"
                    {:type ::inference-checkpoint})))
  (let [{:keys [weights training adam spaces states positions edges iteration]} experiment
        rng (java.util.Random. seed) started (System/nanoTime)
        losses (mapv (fn [_]
                       (let [{:keys [values target t]} (training-example experiment rng)
                             step (swap! iteration inc)
                             loss ((:train-fn training) weights (:m adam) (:v adam)
                                                        values spaces target states positions
                                                        (:src-edges edges) (:dst-edges edges) (double t) 3 (long (:n-edges edges))
                                                        1.0 (double rate) 0.9 0.999 1e-8 (long step))]
                         (when-not (Double/isFinite loss)
                           (throw (ex-info "Nonfinite GSDM loss" {:iteration step :loss loss})))
                         loss)) (range updates))]
    {:updates updates :total-updates @iteration :seed seed :rate rate
     :seconds (/ (- (System/nanoTime) started) 1e9)
     :first-window-loss (/ (reduce + (take 100 losses)) (min 100 updates))
     :last-window-loss (/ (reduce + (take-last 100 losses)) (min 100 updates))}))

(defn numerical-audit
  "Audit the actual fused backward pass without compiling another gradient.
  On fresh weights, an unclipped zero-rate Adam step leaves parameters fixed
  and writes m=(1-beta1)*gradient. Compare those gradients with independent
  finite differences of compiled forward MSE at selected parameter leaves."
  [compiled seed]
  (let [experiment (fresh compiled (get-in compiled [:config :prediction]))
        {:keys [weights training adam spaces states positions edges]} experiment
        {:keys [values target t]} (training-example experiment (java.util.Random. seed))
        loss (fn [] (Math/pow (- (aget ^doubles (predict experiment values t) 0)
                                 (aget ^doubles target 0)) 2))
        forward-loss (loss)
        training-loss ((:train-fn training) weights (:m adam) (:v adam)
                                            values spaces target states positions (:src-edges edges) (:dst-edges edges)
                                            (double t) 3 (long (:n-edges edges)) 1e12 0.0 0.9 0.999 1e-8 1)
        paths [[:embed-We] [:embed-state] [:temb-W1] [:layers 0 :attn-Wq]
               [:layers (dec (get-in compiled [:config :n-layers])) :res-W1]
               [:unembed-Wu2] [:unembed-bu2]]
        checks (vec (for [path paths
                          :let [parameter ^doubles (get-in weights path)
                                moment ^doubles (get-in (:m adam) path)]
                          index (distinct [0 (dec (alength parameter))])
                          :let [original (aget parameter index) h 1e-5
                                derivative (try
                                             (aset parameter index (+ original h))
                                             (let [plus (loss)]
                                               (aset parameter index (- original h))
                                               (/ (- plus (loss)) (* 2 h)))
                                             (finally (aset parameter index original)))
                                gradient (/ (aget moment index) (- 1.0 0.9))]]
                      {:path path :index index :gradient gradient :finite-difference derivative
                       :scaled-error (/ (Math/abs (- gradient derivative))
                                        (max 1.0 (Math/abs gradient) (Math/abs derivative)))}))]
    {:seed seed :timestep t :forward-loss forward-loss :training-loss training-loss
     :loss-error (Math/abs (- forward-loss training-loss))
     :max-scaled-gradient-error (apply max (map :scaled-error checks)) :checks checks}))

(defn denoising-check
  "Held-out denoising MSE versus its analytic conditional-mean optimum.
  Compares predictions on the SAME draws, avoiding training-loss anecdotes."
  [experiment n seed]
  (let [rng (java.util.Random. seed)
        errors (repeatedly n
                           (fn []
                             (let [{:keys [values z t]} (training-example experiment rng)
                                   abar (aget ^doubles (:alpha-bars experiment) t)
                                   precision (+ 3.0 (/ abar (- 1 abar)))
                                   optimal (/ (+ (aget ^doubles values 1) (aget ^doubles values 2)
                                                 (/ (* (Math/sqrt abar) (aget ^doubles values 0)) (- 1 abar))) precision)
                                   predicted (predict-x0 experiment values t)
                                   truth z]
                               {:model (Math/pow (- predicted truth) 2)
                                :oracle (Math/pow (- optimal truth) 2)
                                :excess (Math/pow (- predicted optimal) 2)})))]
    (into {} (for [k [:model :oracle :excess]] [k (/ (reduce + (map k errors)) n)]))))

(defn posterior-draws
  "Conditional DDPM samples of z. Uses the compiled graph network and the
  same fixedlarge reverse-variance convention as Raster's gsdm-sample."
  [experiment {:keys [x y]} n seed]
  (let [rng (java.util.Random. seed) betas ^doubles (:betas experiment)
        bars ^doubles (:alpha-bars experiment)]
    (mapv (fn [_]
            (loop [t (dec (alength betas)) z (.nextGaussian rng)]
              (if (neg? t) z
                  (let [abar (aget bars t) beta (aget betas t) previous (if (pos? t) (aget bars (dec t)) 1.0)
                        prediction (if (:oracle-denoiser? experiment)
                                     (/ (+ x (- y (* 2 x)) (/ (* (Math/sqrt abar) z) (- 1 abar)))
                                        (+ 3.0 (/ abar (- 1 abar))))
                                     (predict-x0 experiment (double-array [z x (- y (* 2 x))]) t))
                        mean (+ (* (/ (* (Math/sqrt previous) beta) (- 1 abar)) prediction)
                                (* (/ (* (Math/sqrt (- 1 beta)) (- 1 previous)) (- 1 abar)) z))
                        next (+ mean (if (pos? t) (* (Math/sqrt beta) (.nextGaussian rng)) 0.0))]
                    (recur (dec t) next))))) (range n))))

(defn summarize [draws]
  (let [mu (/ (reduce + draws) (count draws))
        variance (/ (reduce + (map #(Math/pow (- % mu) 2) draws)) (dec (count draws)))]
    {:mean mu :variance variance :draws (count draws)}))

(defn public-sampler-check
  "Compare Raster's public x0 DDPM sampler with the compiled experiment
  sampler under the SAME RNG stream. Tests implementation agreement and
  evidence clamping, separately from learned posterior accuracy."
  [experiment {:keys [x y] :as evidence} n seed]
  (when-not (= :x0 (get-in experiment [:config :prediction]))
    (throw (ex-info "Raster's public sampler currently expects x0 predictions" {:type ::prediction})))
  (let [rng (java.util.Random. seed) started (System/nanoTime)
        {:keys [flat config spaces states edges betas]} experiment
        samples (mapv (fn [_] (vec (g/gsdm-sample flat config spaces states
                                                  (double-array [0.0 x (- y (* 2 x))])
                                                  (:src-edges edges) (:dst-edges edges)
                                                  3 (:n-edges edges) betas :rng rng))) (range n))
        fast (posterior-draws experiment evidence n seed)]
    {:draws n :seed seed :seconds (/ (- (System/nanoTime) started) 1e9)
     :max-draw-error (apply max (map #(Math/abs (- (first %1) %2)) samples fast))
     :observations-clamped? (every? #(and (= x (second %)) (= (- y (* 2 x)) (nth % 2))) samples)
     :public-sampler-posterior (summarize (mapv first samples))}))

(defn posterior-check [experiment evidence n seed]
  (let [started (System/nanoTime) draws (posterior-draws experiment evidence n seed)
        expected (oracle evidence) stats (summarize draws)
        counterfactuals (mapv #(- (:y evidence) (* 3 %)) draws)]
    {:evidence evidence :reference expected :posterior stats
     :counterfactual-y-do-z-zero (summarize counterfactuals)
     :mean-error (- (:mean stats) (:z-mean expected))
     :variance-ratio (/ (:variance stats) (:z-variance expected))
     :seconds (/ (- (System/nanoTime) started) 1e9)}))

(defn calibration-check
  "Held-out simulator pairs: rank histogram and empirical central coverage.
  Small-run diagnostics, not a formal uniformity certificate. Uses independent
  training/evaluation seeds and retains errors against the analytic posterior."
  [experiment cases draws seed]
  (let [rng (java.util.Random. seed)
        reports (mapv (fn [i]
                        (let [z (.nextGaussian rng) x (+ z (.nextGaussian rng))
                              y (+ (* 2 x) z (.nextGaussian rng)) evidence {:x x :y y}
                              samples (posterior-draws experiment evidence draws (+ seed i 1))
                              sorted (vec (sort samples))
                              lo (sorted (int (* 0.05 (dec draws))))
                              hi (sorted (int (* 0.95 (dec draws))))
                              stats (summarize samples)]
                          {:rank (count (filter #(< % z) samples)) :covered? (<= lo z hi)
                           :mean-error (- (:mean stats) (:z-mean (oracle evidence)))
                           :variance-ratio (/ (:variance stats) (:z-variance (oracle evidence)))})) (range cases))]
    {:cases cases :draws-per-case draws :seed seed
     :rank-histogram (mapv #(get (frequencies (map (fn [{:keys [rank]}] (min 9 (int (* 10 (/ rank (inc draws)))))) reports)) % 0) (range 10))
     :coverage-90 (/ (count (filter :covered? reports)) (double cases))
     :conditional-mean-rmse (Math/sqrt (/ (reduce + (map #(Math/pow (:mean-error %) 2) reports)) cases))
     :mean-variance-ratio (/ (reduce + (map :variance-ratio reports)) cases)}))

(declare causal-query)

(defn check-causal-query!
  "Exercise the actual Foerster CPS world API, comparing every replay with
  the known SCM equation, then report the resulting counterfactual moments."
  [experiment evidence n seed]
  (let [p (promise)]
    (binding [engine/*execution-context* (context/create-execution-context)]
      ((causal-query experiment evidence n seed) #(deliver p [:ok %]) #(deliver p [:error %])))
    (let [[status population] (deref p 60000 [:timeout nil])]
      (when-not (= status :ok)
        (throw (ex-info "Causal replay failed" {:status status} population)))
      (let [errors (for [draw population
                         :let [z (get-in draw [:worlds :factual :z])
                               actual (get-in draw [:worlds :z-zero :y])]]
                     (Math/abs (- actual (- (:y evidence) (* 3 z)))))]
        {:draws (count population) :max-replay-error (apply max errors)
         :complete? (every? #(= :complete (:status %)) population)
         :y-do-z-zero (summarize (mapv #(get-in % [:worlds :z-zero :y]) population))
         :y-do-x-zero (summarize (mapv #(get-in % [:worlds :x-zero :y]) population))}))))

(defn causal-query
  "CPS query through Foerster, from GSDM's equal-weight approximate abductions.
  Do NOT use Raster's INTERVENED embedding as a substitute for causal replay."
  [experiment evidence n seed]
  (cf/query (scm)
            {:abductions (mapv #(abduction evidence %) (posterior-draws experiment evidence n seed))
             :worlds [[:z-zero {:z {:do 0.0}}] [:x-zero {:x {:do 0.0}}]]}))

(defn save! [experiment path report]
  (io/make-parents path)
  (spit path (pr-str {:config (:config experiment) :iteration @(:iteration experiment)
                      :checkpoint-kind :inference-only :proposal-density :unavailable
                      :betas (vec (:betas experiment))
                      :weights (into {} (map (fn [[k v]] [k (vec v)]) (:flat experiment)))
                      :report report})))

(defn restore
  "Restore inference weights using an already compiled matching architecture.
  Optimizer moments are not saved; resuming training is explicitly refused."
  [compiled path]
  (let [{:keys [config weights betas iteration]} (edn/read-string (slurp path))
        shape-keys [:emb-dim :n-layers :n-heads :n-spaces :n-states :timesteps :dense?]]
    (when-not (= (select-keys config shape-keys) (select-keys (:config compiled) shape-keys))
      (throw (ex-info "Checkpoint architecture does not match compiled experiment" {:type ::architecture})))
    (let [restored (fresh compiled (:prediction config))]
      (when-not (= (set (keys weights)) (set (keys (:flat restored))))
        (throw (ex-info "Checkpoint must contain every parameter exactly once" {:type ::weight-keys})))
      (doseq [[k values] weights]
        (let [destination ^doubles (get-in restored [:flat k])]
          (when-not (and destination (= (count values) (alength destination)))
            (throw (ex-info "Checkpoint weight shape mismatch" {:type ::weight-shape :key k})))
          (System/arraycopy (double-array values) 0 destination 0 (alength destination))))
      (reset! (:iteration restored) iteration)
      (assoc restored :betas (double-array betas)
             :alpha-bars (diffusion/compute-alphas-cumprod (double-array betas))
             :inference-only? true))))

(defn -main [& [output]]
  (let [experiment (prepare {:prediction :v :emb-dim 16 :n-layers 2})
        before (denoising-check experiment 256 201)
        training [(train! experiment 100000 {:seed 103 :rate 0.0003})
                  (train! experiment 100000 {:seed 104 :rate 0.00005})]
        after (denoising-check experiment 4096 204)
        checks (mapv #(posterior-check experiment % 2048 304)
                     [{:x 1.0 :y 2.5} {:x -1.0 :y -3.0} {:x 2.0 :y 6.0} {:x 0.0 :y 0.0}])
        report {:preparation-seconds (:preparation-seconds experiment)
                :training training :denoising-before before :denoising-after after :checks checks
                :calibration (calibration-check experiment 64 128 401)
                :causal (check-causal-query! experiment {:x 1.0 :y 2.5} 128 501)}]
    (save! experiment (or output "experiments/results/gsdm/reproduction.edn") report)
    (prn report)
    (shutdown-agents)))
