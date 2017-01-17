::

    Module:

        ./MarkovAgent
             +---- demo.py
                     +---- learn.py
                              +---- encoding.py
                              +---- transitions.py / rewards.py / policy.py

    -------------------------------------------------------------------------------------------

    Markov Agent:

        Reward of each state->action      --+
                                            +---> Policy (rewards, probabilities)
        Probability of each state->action --+

    -------------------------------------------------------------------------------------------

    Goal: find the best policy of each state->action
          (each) state->action: reward / probabilty --> policy (update action_value)
    Steps:
        -> preprocessing: encoding state->action
        -> leanring:
            -> (each) state->action: reward / probablity
            -> (each) state->action: policy (update action_value by iterations)
        -> postprocessing: decoding the policy of each state->action
