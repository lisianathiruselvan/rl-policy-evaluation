# POLICY EVALUATION

## AIM:
To evaluate and compare the performance of two policies using policy evaluation.

## PROBLEM STATEMENT:
Given a set of states, actions, and transition probabilities, we are given two policies.

We want to evaluate the performance of the two policies by computing their state-value functions.

The policy with the higher state-value function is considered to be the better policy.

## POLICY EVALUATION FUNCTION:
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V=np.zeros(len(P))
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V)):
        break
      prev_V=V.copy()
    return V
```
## OUTPUT:

### Policy 1

![image](https://github.com/lisianathiruselvan/rl-policy-evaluation/assets/119389971/ed51c1e8-c665-43cf-a093-0ffe22e71206)


### Policy 2

![image](https://github.com/lisianathiruselvan/rl-policy-evaluation/assets/119389971/049553a1-f4bc-4b66-905f-57c8f7fe3dbd)


## RESULT:

Thus, the evaluation and comparison of the two policies using policy evaluation has been done successfully and it is found that policy 2 is better.
