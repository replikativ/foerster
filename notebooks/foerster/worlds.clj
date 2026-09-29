;; # Models in Worlds

;; A pure model only computes. A model that runs in an application — an agent
;; reasoning about a room, a simulation over a database — also *reads and
;; writes the systems of the world it runs in*. If a thousand particles
;; wrote to the real database, inference would corrupt it. With
;; `:world-policy :fork`, every particle runs in its own frozen copy of the
;; caller's world instead, and nothing it writes reaches the caller.

(ns foerster.worlds
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.yggdrasil :as ygg]
            [yggdrasil.convergent.gset :as g]))

(def world (sp/create-execution-context))
(random/set-seed! 3)

;; ## A system in the world
;;
;; spindel worlds hold *systems* — databases, repositories, stores — through
;; [yggdrasil](https://github.com/replikativ/yggdrasil), which can fork them.
;; Here a small in-memory set stands in for a knowledge base the model
;; consults and extends.

(sp/with-context world
  (ygg/register! (-> (g/gset "notes" {:store-config {:backend :memory :id (random-uuid)}}
                             {:sync? true})
                     (g/conj :known-fact {:sync? true})))
  :registered)

;; `ygg/system` resolves a system by its name in the world bound at the
;; time — in a particle, that particle's fork of it:

(defn note! [x]
  (reset! (ygg/system-signal "notes") (g/conj (ygg/system "notes") x {:sync? true})))

(defn notes [] (g/elements (ygg/system "notes") {:sync? true}))

;; A model that writes its hypothesis into the knowledge base before it is
;; scored — as an agent would record what it is considering:

(defn hypothesis []
  (spin
   (let [h (sample (dist/categorical {:rain 0.3 :sprinkler 0.7}) :id :cause)]
     (note! h)
     (observe (dist/flip (if (= h :rain) 0.9 0.4)) true :id :wet-grass)
     {:cause h :notes (notes)})))

;; ## Forked worlds
;;
;; With `:world-policy :fork`, the model runs in a fork of `world`, and every
;; particle in a fork of that:

(def posterior
  (sp/with-context world
    @(infer/smc-infer (hypothesis) 500 {:world-policy :fork})))

;; Each particle saw the known fact and its own hypothesis, nothing from any
;; other particle:

(frequencies (map (comp :notes m/get-value) (m/get-contexts posterior)))

;; and the caller's knowledge base is unchanged:

(sp/with-context world (notes))

;; The posterior of the cause, P(rain | wet grass) = 0.27 / (0.27 + 0.28):

(infer/query posterior #(if (= :rain (:cause %)) 1.0 0.0))

;; The posterior keeps no world, only each particle's *descriptor*: a
;; portable record of the fork it ran in, discarded before the result was
;; delivered.

(first (m/world-descriptors posterior))

;; ## Every particle runs the whole model
;;
;; In pure inference (`:world-policy :fresh`), the part of a model before its
;; first random choice runs once and is shared. In canonical worlds every
;; particle runs the model from its start: a model's effects may be random
;; without a sample site — a call to a language model, a tool — and sharing
;; that prefix would give all particles the same answer.

(def calls (atom 0))

(defn with-an-effect []
  (spin
   (swap! calls inc)                       ; an effect before any sample site
   (let [x (sample (dist/normal 0.0 1.0) :id :x)]
     (observe (dist/normal x 1.0) 0.5)
     x)))

(do (reset! calls 0)
    (sp/with-context world @(infer/smc-infer (with-an-effect) 100 {:world-policy :fork}))
    @calls)

;; ## Budgets
;;
;; A fork copies state; it must not copy authority to spend. If each of 500
;; particles could spend the room's whole budget, inference would spend 500
;; times it. With a *resource authority* (spindel's
;; `world.scope/PResourceAuthority`) and a `:grant`, the root of the
;; inference draws the grant from the caller's wallet, every particle an
;; even share of it, and every resampled copy an even share of what its
;; ancestor has left. What a world has not spent goes back.
;;
;; A toy ledger: wallets by world, each remembering where it came from.

(defn ledger-authority [ledger]
  (reify world-scope/PResourceAuthority
    (grant! [_ source child grant]
      (swap! ledger (fn [book]
                      (-> book
                          (update-in [(:fork-id source) :wallet] #(merge-with - % grant))
                          (assoc (:fork-id child) {:wallet grant :from (:fork-id source)}))))
      nil)
    (return! [_ context]
      (swap! ledger (fn [book]
                      (if-let [{:keys [wallet from]} (get book (:fork-id context))]
                        (-> book
                            (update-in [from :wallet] #(merge-with + % wallet))
                            (dissoc (:fork-id context)))
                        book)))
      nil)
    (balance [_ context] (get-in @ledger [(:fork-id context) :wallet]))
    (escrow! [_ _ _] nil)
    (claim! [_ _ _] nil)))

(def ledger (atom {(:fork-id world) {:wallet {:tokens 1000}}}))

(defn tokens [] (get-in @ledger [(:fork-id (ec/current-execution-context)) :wallet :tokens]))

(defn budgeted []
  (spin
   (let [budget (tokens)
         x (sample (dist/normal 0.0 1.0) :id :x)]
     (observe (dist/normal x 1.0) 0.5)
     budget)))

(def budgets
  (sp/with-context world
    @(infer/smc-infer (budgeted) 10 {:world-policy :fork
                                     :authority (ledger-authority ledger)
                                     :grant {:tokens 100}})))

;; Ten particles, each started with a tenth of the 100 tokens granted:

(mapv m/get-value (m/get-contexts budgets))

;; and afterwards everything is back in the caller's wallet:

@ledger

;; ## What cannot be copied
;;
;; Particles are copies of worlds, so the *grades* of the world's systems
;; apply (`ygg/register!` `:grade`). A system that holds a live handle — an
;; open connection, a device — is `:affine`: it may be discarded but never
;; copied. A world holding one is refused before the model runs:

(def with-a-live-handle (sp/create-execution-context))

(sp/with-context with-a-live-handle
  (ygg/register! (g/gset "connection" {:store-config {:backend :memory :id (random-uuid)}}
                         {:sync? true})
                 {:grade :affine})
  :registered)

(try
  (sp/with-context with-a-live-handle
    @(infer/smc-infer (hypothesis) 10 {:world-policy :fork}))
  (catch Exception e
    {:error (ex-message e)
     :cause (some-> e ex-cause ex-data :type)}))

;; Systems that settle by replaying intents — a business ledger whose legal
;; numbers are assigned when a world is merged — may be copied: see spindel's
;; [forking guide](https://github.com/replikativ/spindel/blob/main/docs/forking.md).
;;
;; ## Failure
;;
;; However inference ends — a result, a failure of the model, or the
;; cancellation of the inference spin — every world is discarded before the
;; outcome is delivered. A failure carries `:world/recovery` in its data:
;; the worlds' descriptors and operations to finish their cleanup, should the
;; automatic one have failed. See the
;; [worlds guide](https://github.com/replikativ/foerster/blob/main/doc/worlds.md).
