(ns org.replikativ.foerster.runtime.lifetime
  "The lifetime of one inference: what cancelling it stops.

  Every root world an inference creates (a session's root, a chain's world,
  the frozen fork of the caller's world) is created through `world` or
  `fork`, which register it here. Cancelling the lifetime releases them:
  every world forked from a root goes with it (engine v2 F2), the model
  runs in them stop, and a driver waiting on one of them settles with
  `::released` (savepoint ends, `async/op-of`). A root created after the
  cancellation is released at once and refused, and `check!` throws, so a
  driver stops at its next await or loop head.

  The lifetime travels in the inference's options (`::lifetime`), which a
  driver forwards to the methods it runs, so they share it. It is created
  by `canonical/with-frozen-caller`, the one entry of every public method,
  which also connects it to the cancellation of the task awaiting it
  (`async/external-of`) and finishes it when the inference ends.

  Roots are held weakly: a long inference creating many short-lived roots
  keeps none of them alive."
  (:require [org.replikativ.spindel.v2.core :as v2]
            [org.replikativ.spindel.v2.host :as hs]))

(defn create [] (atom {:state :running :roots [] :prune 8}))

(defn of
  "The lifetime `opts` name, or nil."
  [opts]
  (::lifetime opts))

(defn with
  "`opts` naming lifetime `l`."
  [opts l]
  (assoc opts ::lifetime l))

(defn cancelled-ex []
  (v2/cancelled-ex ::cancelled))

(defn- stop? [e]
  (and e (or (v2/cancelled? e) (:spindel/released (ex-data e)))))

(defn stopping?
  "Whether error `e` (or one of its causes) means the inference was
  cancelled or its worlds released: never a model failure, a rejected
  proposal or a particle weight, and never retried."
  [e]
  (boolean (some stop? (take-while some? (iterate ex-cause e)))))

(defn cancelled?
  "Whether lifetime `l` was cancelled."
  [l]
  (= :cancelled (:state @l)))

(defn check!
  "Throws once the lifetime `opts` name is cancelled: a driver's loop head
  that awaits nothing in its worlds."
  [opts]
  (when-let [l (of opts)]
    (when (cancelled? l) (throw (cancelled-ex)))))

(defn- prune [roots]
  (filterv hs/deref-weak roots))

(defn own!
  "Registers root world `w` with the lifetime `opts` name; once it is
  cancelled, releases `w` and throws instead. Returns `w`."
  [opts w]
  (when-let [l (of opts)]
    (let [[before _] (swap-vals! l (fn [{:keys [state roots] :as s}]
                                     (if (= :running state)
                                       (let [roots (conj roots (hs/weak w))]
                                         (if (> (count roots) (:prune s))
                                           (let [roots (prune roots)]
                                             (assoc s :roots roots :prune (max 8 (* 2 (count roots)))))
                                           (assoc s :roots roots)))
                                       s)))]
      (when (= :cancelled (:state before))
        (v2/release-world! w)
        (throw (cancelled-ex)))))
  w)

(defn world
  "A new root world (`v2/world` with `world-opts`), owned by the lifetime
  `opts` name."
  [opts world-opts]
  (own! opts (if world-opts (v2/world world-opts) (v2/world))))

(defn fork
  "A host fork of `source` (`v2/fork!`), owned by the lifetime `opts` name."
  [opts source]
  (own! opts (v2/fork! source)))

(defn cancel!
  "Cancels lifetime `l`: releases every root it holds. Once only; does
  nothing after the inference finished."
  [l]
  (let [[before _] (swap-vals! l (fn [s] (if (= :running (:state s)) (assoc s :state :cancelled :roots []) s)))]
    (when (= :running (:state before))
      (doseq [r (:roots before)]
        (when-let [w (hs/deref-weak r)]
          (try (v2/release-world! w) (catch #?(:clj Throwable :cljs :default) _ nil)))))
    nil))

(defn finish!
  "The inference that created `l` ended: later cancellation does nothing,
  and the roots it made are the driver's (or its result's) to give back."
  [l]
  (swap! l (fn [s] (if (= :running (:state s)) (assoc s :state :finished :roots []) s)))
  nil)

(defn released-end?
  "Whether a savepoint layer's terminal `event` is the end of a world its
  owner released (`::released` at the abandoned site): the inference was
  cancelled, unlike a particle its own driver abandoned."
  [event]
  (boolean (:spindel/released (ex-data (:savepoint/payload event)))))

(defn carry
  "`to` (options a driver builds for a method it runs) naming the lifetime
  `from` names, if any."
  [to from]
  (if-let [l (of from)] (with to l) to))
