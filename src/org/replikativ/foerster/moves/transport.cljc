(ns org.replikativ.foerster.moves.transport
  "Scored forward/backward transport. The caller supplies valid auxiliary
  proposal laws and the change-of-variables Jacobian; these are not inferred.
  The Jacobian must be stated (`:log-jacobian`, 0.0 for a discrete or
  volume-preserving map): a forgotten one would silently bias the weights."
  (:require [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.runtime.trace :as it]
            [org.replikativ.foerster.moves.pair :as pair]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.trace :as trace])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defrecord Kernel [source destination forward backward])

(defn kernel
  "Bind forward/backward proposal laws to two targets: a pair
  (`foerster.moves.pair`), or its two kernels. Value forward returns
  {:value y :log-q ... :reverse u :log-jacobian ...}; trace forward receives
  [choices trace] and returns :updates instead of :value, as a pair's
  does. Backward scores u at y (or its choice map). Forward and reverse
  support must match."
  ([source destination move]
   (pair/check! move "A transport kernel")
   (kernel source destination (:forward move) (:backward move)))
  ([source destination forward backward]
   (when-not (and (ifn? forward) (ifn? backward))
     (throw (ex-info "Transport needs scored forward/backward proposals" {:type ::unscored-proposal})))
   (->Kernel (:id source) (:id destination) forward backward)))

(defn require-jacobian!
  "Refuse a move `proposed` (a forward kernel's or an involution's result)
  that does not state its `:log-jacobian`. `what` names the move."
  [proposed what]
  (when-not (or (number? (:log-jacobian proposed))
                ;; staying put: no map, no Jacobian
                (and (contains? proposed :updates) (empty? (:updates proposed))))
    (throw (ex-info (str what " must return :log-jacobian, the log |det| of its "
                         "deterministic map (0.0 for a discrete or volume-preserving move)")
                    {:type ::missing-jacobian :keys (vec (keys proposed))})))
  proposed)

(defn correction [log-target-ratio log-forward log-backward log-jacobian]
  (let [terms [log-target-ratio log-forward log-backward (or log-jacobian 0.0)]]
    (when-not (every? #(and (number? %) (not (#?(:clj Double/isNaN :cljs js/isNaN) %))) terms)
      (throw (ex-info "Transport returned an invalid score" {:type ::invalid-score})))
    (let [result (+ log-target-ratio (- log-forward) log-backward (or log-jacobian 0.0))]
      (when (#?(:clj Double/isNaN :cljs js/isNaN) result)
        (throw (ex-info "Transport correction is indeterminate" {:type ::invalid-score})))
      result)))

(defn value-step [source destination kernel copy x]
  (let [old-score ((:score source) x)
        {:keys [value log-q reverse log-jacobian] :as proposed} ((:forward kernel) (copy x))]
    (when-not (contains? proposed :value)
      (throw (ex-info "A value transport must return :value" {:type ::invalid-proposal})))
    (require-jacobian! proposed "A transport's forward kernel")
    {:particle value :log-increment (correction (- ((:score destination) value) old-score)
                                                log-q ((:backward kernel) value reverse) log-jacobian)}))

(defn trace-step [source destination kernel old seed]
  (pcps/async
   (let [{:keys [updates log-q reverse log-jacobian]} (require-jacobian!
                                                       (random/with-stream* seed ::forward #((:forward kernel) (it/choices old) old))
                                                       "A transport's forward kernel")
         unknown (remove #(contains? (:trace/entries old) %) (keys updates))
         _ (when (seq unknown)
             (throw (ex-info "Trace transport cannot update absent sites"
                             {:type ::unknown-sites :addresses (vec unknown)})))
         fixed (filter #(let [note (:note (get-in old [:trace/entries %]))]
                          (or (:observed? note) (:constrained? note))) (keys updates))]
     (when (seq fixed)
       (throw (ex-info "Transport cannot update observed or constrained sites" {:type ::fixed-site :addresses (vec fixed)})))
     (let [next (if (empty? updates) old
                    (pcps/await
                     (it/reexecute old (set (keys updates))
                                   (it/policy {:keep? true :temperature (:position destination)
                                               :draw (fn [site _]
                                                       (let [address (:savepoint/address site)]
                                                         (when (contains? updates address)
                                                           {:value (get updates address) :symmetric? true})))})
                                   {:anchor? it/anchor? :seed seed})))
           failed? (some? (:trace/error next))
           ;; The GFI ratio also accounts for fresh/discarded internal prior
           ;; draws when replay changes control flow. The bridge adds any target
           ;; change at the old state, including factors upstream of replay.
           ratio (cond failed? ##-Inf
                       (identical? next old) (- ((:score destination) old) ((:score source) old))
                       :else (+ (it/mh-log-ratio old next (constantly 0.0) (::it/restart next)
                                                 (it/tempered-score (:position destination)) (::it/changed next))
                                (- ((:score destination) old) ((:score source) old))))
           increment (if failed? ##-Inf
                         (correction ratio log-q ((:backward kernel) (it/choices next) reverse) log-jacobian))]
       (when failed? (trace/release! next old))
       {:particle (if failed? old next) :log-increment increment}))))
