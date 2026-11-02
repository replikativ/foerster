(ns org.replikativ.foerster.write-state-test
  "Incremental reuse and world state a model writes (`itrace/wrote-state?`):
  an incremental move must give the trace a from-scratch run of the same
  choices gives — the same score and the same result."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.canonical :as canonical]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.effects.track :refer [track]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.keyed :refer [keyed]]))

(defn- await* [op]
  (let [p (promise)]
    (op #(deliver p [:ok %]) #(deliver p [:error %]))
    (let [[k v] (deref p 30000 [:timeout nil])]
      (case k
        :ok v
        :error (throw v)
        (throw (ex-info "timed out" {}))))))

(defmacro ^:private in-root [& body]
  `(let [root# (context/create-execution-context)]
     (try
       (binding [ec/*execution-context* root#] ~@body)
       (finally (context/stop-context! root#)))))

(defn- choices [t]
  (into {} (keep (fn [e] (when (and (= itrace/choose-site (:site e)) (not (:observed? (:note e))))
                           [(:address e) (:value e)])))
        (itrace/entries t)))

(defn- move
  "A spin: generate `model` with `init`, move `address` to `v` incrementally
  (`gfi/update`, i.e. `itrace/reexecute`), and assess the moved choices from
  scratch; `opts` are the runs' options."
  [model init address v opts]
  (spin
   (let [t (:trace (aw/await (gfi/generate (model) init opts)))
         {t' :trace} (aw/await (gfi/update t {address v}))
         fresh (aw/await (gfi/assess (model) (choices t') opts))
         out {:wrote? (itrace/wrote-state? t)
              :path (if-let [from (::itrace/restart t')] [:replay from] :keyed-update)
              :result (:trace/result t') :log-joint (itrace/log-joint t')
              :fresh-result (:result fresh) :fresh-log-joint (:weight fresh)}]
     (aw/await (gfi/close! t'))
     out)))

(defn- move! [model init address v]
  (deref (move model init address v nil) 30000 ::timed-out))

(defn- exact? [{:keys [result log-joint fresh-result fresh-log-joint]}]
  (and (= result fresh-result) (< (Math/abs (- log-joint fresh-log-joint)) 1e-9)))

;; -----------------------------------------------------------------------------
;; Models
;; -----------------------------------------------------------------------------

(defn- appending-keys
  "Keys append their value to a shared atom; the parent returns what it
  holds, sorted (keys run in no fixed order). With `prefix?` a site precedes
  the keyed region (an anchor before it). A re-run key that found its old
  append still in place would leave three values."
  [prefix?]
  (fn []
    (spin
     (let [log (sa/atom [])
           m (if prefix? (sample (dist/normal 0.0 1.0) :id :m) 0.0)]
       (aw/await (keyed :jobs [:a :b]
                        (fn [k]
                          (spin
                           (let [x (sample (dist/normal m 1.0) :id [:x k])]
                             (swap! log conj x)
                             (observe (dist/normal x 0.1) 1.0 :id [:y k])
                             x)))))
       (vec (sort @log))))))

(defn- slot-keys
  "Keys write their own slot and read it back. (Reading a sibling's slot is
  a race: keyed children run in no fixed order.)"
  []
  (spin
     (let [slots (sa/atom {})
           m (sample (dist/normal 0.0 1.0) :id :m)]
       (aw/await (keyed :jobs [:a :b]
                        (fn [k]
                          (spin
                           (let [x (sample (dist/normal m 1.0) :id [:x k])]
                             (swap! slots assoc k x)
                             (observe (dist/normal (get @slots k) 0.1)
                                      1.0 :id [:y k])
                             x)))))
       @slots)))

(defn- summary
  "A site-free sub-run sums an atom the model appended to."
  []
  (spin
   (let [log (sa/atom [])
         a (sample (dist/normal 0.0 1.0) :id :a)
         _ (swap! log conj a)
         b (sample (dist/normal 0.0 1.0) :id :b)
         _ (swap! log conj b)
         total (aw/await (spin (reduce + @log)))]
     (observe (dist/normal total 0.1) 1.5 :id :y)
     {:log @log :total total})))

(defn- pure-keys []
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         xs (aw/await (keyed :jobs [0 1 2]
                             (fn [i]
                               (spin (let [x (sample (dist/normal m 1.0) :id [:x i])]
                                       (observe (dist/normal x 0.5) (* 0.5 i) :id [:y i])
                                       x)))))]
     (vec xs))))

;; -----------------------------------------------------------------------------
;; Fresh worlds
;; -----------------------------------------------------------------------------

(deftest detects-written-state
  (in-root
   (let [pure (:trace (await* (gfi/generate (pure-keys) {})))
         writer (:trace (await* (gfi/generate (summary) {})))]
     (try
       (is (not (itrace/wrote-state? pure)))
       (is (itrace/wrote-state? writer))
       (finally (await* (gfi/close! pure)) (await* (gfi/close! writer)))))))

(deftest a-pure-keyed-model-keeps-its-keyed-updates
  (in-root
   (let [r (move! pure-keys {:m 0.2 [:x 0] 0.1 [:x 1] 0.4 [:x 2] 0.9} [:x 1] 0.7)]
     (is (= :keyed-update (:path r)))
     (is (exact? r) (pr-str r)))))

(deftest a-key-that-writes-state-is-not-re-run-on-its-old-writes
  (testing "with an anchor before the keys, the move replays from it"
    (in-root
     (let [r (move! (appending-keys true) {:m 0.0 [:x :a] 0.3 [:x :b] 1.2} [:x :a] 0.9)]
       (is (:wrote? r))
       (is (= [:replay :m] (:path r)))
       (is (= [0.9 1.2] (:result r)) "each key's write once, in program order")
       (is (exact? r) (pr-str r)))))
  (testing "without one, the move is refused rather than wrong"
    (in-root
     (let [e (try (move! (appending-keys false) {[:x :a] 0.3 [:x :b] 1.2} [:x :a] 0.9)
                  (catch Exception e e))]
       (is (= ::itrace/stateful-keyed-move (:type (ex-data e))) (pr-str (ex-data e)))))))

(deftest a-key-that-writes-its-own-slot
  (in-root
   (let [r (move! slot-keys {:m 0.0 [:x :a] 0.3 [:x :b] 1.2} [:x :a] 0.9)]
     (is (exact? r) (pr-str r)))))

(deftest a-sub-run-that-read-an-atom-is-not-adopted-stale
  (in-root
   (doseq [address [:a :b]]
     (let [r (move! summary {:a 0.3 :b 1.2} address 0.9)]
       (is (exact? r) (str address " " (pr-str r)))))))

;; -----------------------------------------------------------------------------
;; Forks of the caller's world
;; -----------------------------------------------------------------------------

(defn- in-frozen-caller
  "The spin `(f opts)` run inside one frozen fork of the caller's world
  (`:world-policy :fork`), `opts` the runs' options; its value."
  [f]
  (deref (canonical/with-frozen-caller {:world-policy :fork}
           (fn [opts] (f (canonical/run-options opts))))
         30000 ::timed-out))

(deftest reads-of-the-caller-s-state-keep-incremental-reuse
  (in-root
   (let [offset (sa/atom 0.5)
         model (fn []
                 (spin
                  (let [m (sample (dist/normal 0.0 1.0) :id :m)
                        xs (aw/await (keyed :jobs [0 1 2]
                                            (fn [i]
                                              (spin (let [x (sample (dist/normal (+ m @offset) 1.0) :id [:x i])]
                                                      (observe (dist/normal x 0.5) (* 0.5 i) :id [:y i])
                                                      x)))))]
                    (vec xs))))
         r (in-frozen-caller #(move model {:m 0.2 [:x 0] 0.1 [:x 1] 0.4 [:x 2] 0.9} [:x 1] 0.7 %))]
     (is (not (:wrote? r)))
     (is (= :keyed-update (:path r)) "keyed reuse is back for a model that only reads")
     (is (exact? r) (pr-str r))
     (is (= 0.5 @offset)))))

(deftest writes-under-fork-re-run-from-the-start
  (in-root
   (let [r (in-frozen-caller #(move (appending-keys false) {[:x :a] 0.3 [:x :b] 1.2} [:x :a] 0.9 %))]
     (is (:wrote? r))
     (is (= :replay (first (:path r))) "from the start site: no anchor precedes the keys")
     (is (= [0.9 1.2] (:result r)))
     (is (exact? r) (pr-str r)))))

;; -----------------------------------------------------------------------------
;; A Spindel limit, recorded: a signal the model writes and tracks
;; -----------------------------------------------------------------------------
;;
;; A computation that writes a signal and then reads it with `track` in a
;; sub-run is re-run once the write is published, although it read the new
;; value already; its awaiting parent resumes again and reaches its next
;; sample site twice, which a trace refuses ("Duplicate site address").
;; Signals are reactive inputs; a model's own state belongs in atoms.

(deftest a-signal-the-model-writes-and-tracks-is-refused-by-the-trace
  (in-root
   (let [model (fn []
                 (spin
                  (let [log (sp/signal [])
                        a (sample (dist/normal 0.0 1.0) :id :a)
                        _ (swap! log conj a)
                        total (aw/await (spin (reduce + (:new (track log)))))]
                    (observe (dist/normal total 0.1) 1.5 :id :y)
                    total)))
         e (try (await* (gfi/generate (model) {:a 0.3})) nil
                (catch Exception e e))]
     (is (some? e) "if this starts passing, Spindel no longer re-runs a reader on its own write"))))
