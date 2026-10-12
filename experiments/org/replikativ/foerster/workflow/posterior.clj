(ns org.replikativ.foerster.workflow.posterior
  "A Bayesian-workflow environment for dvergr agents: posterior inference on a
  posteriordb posterior, verified against its reference draws.

  The candidate gets the Stan program and the data in its workspace and must
  write posterior draws of every reference parameter to `draws-path`, as EDN
  {\"param\" [draw …]}. It may compute them any way it can in its sandbox;
  HTTP is offline, so posteriordb's reference draws are out of reach. They stay
  in the evaluator, which compares
    - each parameter's mean (within `posteriordb/mean-tolerance` reference
      sds) and sd (within 1 ± `posteriordb/sd-tolerance`): the tolerances
      foerster's own samplers are held to;
    - its 5, 25, 50, 75 and 95% quantiles (within `quantile-tolerance`
      reference sds): the shape, not only two moments;
    - the correlation of every pair of parameters (within
      `correlation-tolerance`): the draws are joint draws, index by index.
  The reward is 1 when every check passes.

  A provider in the sense of dvergr's doc/benchmarks.md: `definition`,
  `world-setup` and `evaluator` per posterior."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [dvergr.agent.environment :as environment]
            [dvergr.agent.evaluation :as evaluation]
            [dvergr.sandbox.ns.io :as sandbox-io]
            [dvergr.substrate.geschichte :as g]
            [hasch.core :as hasch]
            [muschel.fs :as fs]
            [org.replikativ.foerster.posteriordb :as pdb]
            [org.replikativ.spindel.engine.core :as ec]))

(def posteriors
  "The posteriordb posteriors foerster's profile covers, all with reference draws."
  ["eight_schools-eight_schools_noncentered" "kidiq-kidscore_momiq" "earnings-logearn_height"
   "mesquite-logmesquite" "sblrc-blr" "arK-arK" "garch-garch11"
   "low_dim_gauss_mix-low_dim_gauss_mix" "dogs-dogs" "wells_data-wells_dist"])

(def model-path "/task/model.stan")
(def data-path "/task/data.json")
(def draws-path "/out/draws.edn")
(def min-draws 1000)
(def max-draws-bytes (* 4 1024 1024))
(def quantile-levels [0.05 0.25 0.5 0.75 0.95])
(def quantile-tolerance 0.3)
(def correlation-tolerance 0.2)

;; --- the reference ----------------------------------------------------------

(defn- mean ^double [xs] (/ (reduce + xs) (count xs)))

(defn- sd ^double [xs]
  (let [mu (mean xs)] (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) xs)) (dec (count xs))))))

(defn- quantiles [xs]
  (let [v (vec (sort xs)) n (count v)]
    (mapv (fn [p] (let [h (* p (dec n)) lo (long (Math/floor h)) hi (min (dec n) (inc lo))]
                    (+ (v lo) (* (- h lo) (- (v hi) (v lo))))))
          quantile-levels)))

(defn- correlation [xs ys]
  (let [mx (mean xs) my (mean ys)
        cov (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys))]
    (/ cov (Math/sqrt (* (reduce + (map #(let [e (- % mx)] (* e e)) xs))
                         (reduce + (map #(let [e (- % my)] (* e e)) ys)))))))

(defn- pairs [params] (for [[i p] (map-indexed vector params) q (drop (inc i) params)] [p q]))

(defn describe
  "{:params [name …] :stats {param {:mean :sd :quantiles}} :correlations
  {[p q] ρ}} of joint draws {param [draw …]}."
  [draws]
  (let [params (vec (sort (keys draws)))]
    {:params params
     :stats (into (sorted-map) (for [p params :let [xs (get draws p)]]
                                 [p {:mean (mean xs) :sd (sd xs) :quantiles (quantiles xs)}]))
     :correlations (into (sorted-map) (for [[p q] (pairs params)]
                                        [[p q] (correlation (get draws p) (get draws q))]))}))

(defn- inputs
  "What one environment is made of, fetched once: the Stan program, the data
  and the reference summary, with their content id."
  [posterior]
  (let [model (pdb/stan-program posterior)
        data (slurp (pdb/fetch-file (str "data/data/" (get (pdb/posterior-info posterior) "data_name") ".json.zip")))
        reference (describe (pdb/reference-draws posterior))]
    {:model model :data data :reference reference
     :basis (hasch/uuid {:posterior posterior :posteriordb/commit pdb/commit
                         :model (hasch/uuid model) :data (hasch/uuid data) :reference (hasch/uuid reference)
                         :draws-path draws-path :min-draws min-draws
                         :tolerances [pdb/mean-tolerance pdb/sd-tolerance quantile-levels
                                      quantile-tolerance correlation-tolerance]})}))

(def ^:private inputs* (memoize inputs))

(defn- task [params]
  (str "Bayesian inference task. " model-path " is a Stan program and " data-path
       " its data (JSON). Compute the posterior distribution of its parameters "
       "and write posterior draws to " draws-path " as an EDN map from parameter name "
       "to a vector of at least " min-draws " numbers, for exactly these parameters: "
       (str/join ", " params) ". Indexed parameters are named as in Stan output, e.g. "
       "\"theta[1]\". Draws must be on the parameters' own (constrained) scale. "
       "Every parameter needs the same number of draws, and the draws at one index must "
       "form one joint draw of all parameters. "
       "You have no Stan installation and no network access; implement the inference "
       "yourself in clojure_eval. The draws are checked against a reference posterior: "
       "every parameter's mean, standard deviation and quantiles, and the correlations "
       "between parameters, must match it closely."))

(defn- environment-id [posterior] (keyword "posteriordb" posterior))

(defn definition
  "The EnvironmentDef of `posterior`."
  [posterior]
  (let [{:keys [reference] b :basis} (inputs* posterior)
        id (environment-id posterior)]
    (environment/make-environment
     {:id id :task (task (:params reference))
      :verifier {:id id :version 1 :basis b}
      :world {:isolation :ctx :settlement :discard
              :setup {:setup/id id :setup/version 1 :setup/basis b}}
      :limits {:timeout-ms 1800000 :cancel-timeout-ms 10000}
      :metadata {:posteriordb/posterior posterior :posteriordb/commit pdb/commit
                 :draws/path draws-path :draws/min min-draws}})))

(def ^:private offline-id (hasch/uuid [:workflow/offline-http-v1]))

(defn world-setup
  "Seeds the Stan program and the data; HTTP answers 404."
  [posterior]
  (let [{:keys [model data basis]} (inputs* posterior)]
    (evaluation/make-world-setup
     {:id (environment-id posterior) :version 1 :basis basis
      :prepare
      (fn [{:keys [room]}]
        (binding [ec/*execution-context* (:ctx room)]
          (sandbox-io/install-http-fixture!
           {:id offline-id :env {}
            :transport (fn [_] {:status 404 :headers {} :body "Offline workflow environment"})})
          (let [filesystem (or (g/filesystem)
                               (throw (ex-info "The environment needs a registered Geschichte workspace" {})))]
            (doseq [[path text] [[model-path model] [data-path data]]]
              (when-not (fs/write-string! filesystem path text false)
                (throw (ex-info "Cannot seed the workflow environment" {:path path}))))))
        {:posteriordb/posterior posterior})})))

(defn- capture
  "The draws file from the candidate's workspace, bounded by `max-draws-bytes`."
  [room]
  (if-not (:ctx room)
    {:status :world-unavailable}
    (binding [ec/*execution-context* (:ctx room)]
      (if-let [filesystem (g/filesystem)]
        (let [{:keys [type size]} (fs/stat filesystem draws-path)]
          (cond
            (nil? type) {:status :missing}
            (not= :file type) {:status :not-file}
            (not (and (integer? size) (<= 0 size max-draws-bytes))) {:status :size-rejected}
            :else (let [text (fs/read-file filesystem draws-path)]
                    (if (string? text) {:status :ok :text text} {:status :unreadable}))))
        {:status :no-workspace}))))

(defn- nesting
  "The deepest nesting of brackets in `text`, outside strings and comments."
  [^String text]
  (loop [i 0 depth 0 top 0 in-string? false]
    (if (>= i (.length text))
      top
      (let [c (.charAt text i)]
        (cond
          in-string? (cond (= c \\) (recur (+ i 2) depth top true)
                           (= c \") (recur (inc i) depth top false)
                           :else (recur (inc i) depth top true))
          (= c \") (recur (inc i) depth top true)
          (= c \;) (let [nl (.indexOf text "\n" (int i))]
                     (recur (if (neg? nl) (.length text) (inc nl)) depth top false))
          (#{\[ \{ \(} c) (recur (inc i) (inc depth) (max top (inc depth)) false)
          (#{\] \} \)} c) (recur (inc i) (dec depth) top false)
          :else (recur (inc i) depth top false))))))

(defn parse-draws
  "{param [double …]} from `text`, or nil: exactly one EDN map from strings to
  vectors of numbers, nothing nested deeper, no reader tags, nothing after it."
  [text]
  (when (and (string? text) (<= (nesting text) 2))
    (let [v (try
              (with-open [r (java.io.PushbackReader. (java.io.StringReader. text))]
                (let [eof (Object.)
                      opts {:readers {} :default (fn [_ _] (throw (ex-info "tagged" {}))) :eof eof}
                      v (edn/read opts r)]
                  (when (identical? eof (edn/read opts r)) v)))
              (catch Throwable _ nil))]
      (when (and (map? v) (every? string? (keys v)) (every? vector? (vals v))
                 (every? #(every? number? %) (vals v)))
        (into {} (map (fn [[k xs]] [k (mapv double xs)])) v)))))

(defn check-draws
  "The checks a candidate's `draws` ({param [double …]} or nil) can fail
  against `reference` (`describe` of the reference draws), and per parameter
  and pair what it got."
  [reference draws]
  (let [params (:params reference)
        usable? (and (some? draws)
                     (= (set params) (set (keys draws)))
                     (apply = (map count (vals draws)))
                     (>= (count (val (first draws))) min-draws)
                     (every? #(every? (fn [x] (Double/isFinite x)) %) (vals draws)))
        got (when usable? (describe draws))
        per (into (sorted-map)
                  (for [p params :when got
                        :let [{ref-mean :mean ref-sd :sd ref-q :quantiles} (get-in reference [:stats p])
                              {:keys [mean sd quantiles]} (get-in got [:stats p])]]
                    [p {:z (/ (- mean ref-mean) ref-sd)
                        :sd-ratio (/ sd ref-sd)
                        :quantile-error (reduce max (map #(Math/abs (/ (- %1 %2) ref-sd)) quantiles ref-q))}]))
        corr (into (sorted-map)
                   (for [[pq rho] (:correlations reference) :when got]
                     [pq (Math/abs (- (get-in got [:correlations pq]) rho))]))
        all? (fn [pred xs] (and usable? (every? pred xs)))]
    {:checks {:draws? (some? draws)
              :all-params? (and (some? draws) (= (set params) (set (keys draws))))
              :joint-draws? usable?
              :means? (all? #(< (Math/abs (:z %)) pdb/mean-tolerance) (vals per))
              :sds? (all? #(< (Math/abs (- (:sd-ratio %) 1.0)) pdb/sd-tolerance) (vals per))
              :quantiles? (all? #(< (:quantile-error %) quantile-tolerance) (vals per))
              :correlations? (all? #(< % correlation-tolerance) (vals corr))}
     :params per
     :correlation-errors corr}))

(defn evaluator
  "Verifies the captured draws against the reference posterior."
  [posterior]
  (let [{:keys [reference basis]} (inputs* posterior)]
    (evaluation/make-evaluator
     {:id (environment-id posterior) :version 1 :basis basis
      :capture (fn [{:keys [world/room]}] (capture room))
      :observe (fn [{:keys [default execution/evidence result]}]
                 (assoc default :draws evidence
                        :completed? (= :completed (:run/status result))))
      :verify (fn [_ evidence]
                (let [{:keys [status text]} (:draws evidence)
                      {:keys [checks params correlation-errors]}
                      (check-draws reference (when (= :ok status) (parse-draws text)))
                      checks (assoc checks :completed? (true? (:completed? evidence)))]
                  {:checks checks
                   :params params
                   :correlation-errors correlation-errors
                   :reward (if (every? true? (vals checks)) 1.0 0.0)}))})))
