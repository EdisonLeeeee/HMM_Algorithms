import numpy as np
from hmmlearn import hmm
np.random.seed(42)
np.set_printoptions(precision=3)


def HMMfwd(obs, pi, a, b):
    '''Forward Algorithm
    
    Parameters
    ----------
    obs: observation sequence [o_1, o_2, ..., o_T]  
        (T, )    
    pi: start probity   
        (n_states,)
    a: transition matrix  
        (n_statrs, n_states)
    b: emission matrix 
        (n_states, n_obs)

        
    Returns
    -------
    alpha: Forward probability matrix
        (n_states, T)
    '''

    n_states = np.shape(b)[0]
    T = np.shape(obs)[0]

    alpha = np.zeros((n_states, T))
    alpha[:, 0] = pi * b[:, obs[0]]

    for t in range(1, T):
        alpha[:, t] = np.sum(a * alpha[:, t - 1].reshape(-1, 1),
                             axis=0) * b[:, obs[t]]

    return alpha


def HMMbwd(obs, a, b):
    '''Backward Algorithm
    
    Parameters
    ----------
    obs: observation sequence [o_1, o_2, ..., o_T]  
        (T, )    
    a: transition matrix  
        (n_statrs, n_states)
    b: emission matrix 
        (n_states, n_obs)

        
    Returns
    -------
    beta: Backward probability matrix
        (n_states, T)
    '''
    n_states = np.shape(b)[0]
    T = np.shape(obs)[0]

    beta = np.zeros((n_states, T))

    beta[:, -1] = 1.0

    for t in range(T - 2, -1, -1):
        for s in range(n_states):
            beta[:, t] = np.sum(a[s, :] * b[:, obs[t + 1]] * beta[:, t + 1])

    return beta


def Viterbi(obs, pi, a, b):
    '''Viterbi Algorithm
    
    Parameters
    ----------
    obs: observation sequence [o_1, o_2, ..., o_T]  
        (T, )
    pi: start probity   
        (n_states,)
    a: transition matrix  
        (n_statrs, n_states)
    b: emission matrix 
        (n_states, n_obs)
        
    Returns
    -------
    path: possible state sequence
        (n_states, )
    '''

    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    path = np.zeros(T, dtype=np.int32)
    delta = np.zeros((nStates, T))
    phi = np.zeros((nStates, T))

    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])

    path[-1] = np.argmax(delta[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t + 1], t + 1]

    return path


def BaumWelch(obs,
              n_states,
              n_obs,
              pi=None,
              a=None,
              b=None,
              tol=1e-2,
              n_iter=10):
    '''BaumWelch Algorithm
    
    Parameters
    ----------
    obs: observation sequence [o_1, o_2, ..., o_N]  
        (T, )
    n_states: number of states
        int
    n_obs: number of observations
        int
    pi: start probity   
        (n_states,)
    a: transition matrix  
        (n_statrs, n_states)
    b: emission matrix 
        (n_states, n_obs)
    tol: tolerance
        float
    n_iter: max iteration to stop
        int
        
    Returns
    -------
    pi, a, b
    '''
    T = np.shape(obs)[0]
    # initialize
    if pi is None:
        pi = np.ones((n_states)) / n_states
    if a is None:
        a = np.random.rand(n_states, n_states)
    if b is None:
        b = np.random.rand(n_states, n_obs)
    xi = np.zeros((n_states, n_states, T))

    nits = 0
    while True:
        nits += 1
        old_a = a.copy()
        old_b = b.copy()
        alpha = HMMfwd(
            obs,
            pi,
            a,
            b,
        )
        beta = HMMbwd(obs, a, b)
        gamma = alpha * beta
        gamma /= gamma.sum(0)

        # E-step
        for t in range(T - 1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[i, j,
                       t] = alpha[i, t] * beta[j, t +
                                               1] * a[i, j] * b[j, obs[t + 1]]
            xi[:, :, t] /= xi[:, :, t].sum()

        # The last step has no b, beta in
        for i in range(n_states):
            for j in range(n_states):
                xi[i, j, -1] = alpha[i, -1] * a[i, j]
        xi[:, :, -1] /= xi[:, :, -1].sum()

        # M-step
        for i in range(n_states):
            for j in range(n_states):
                a[i, j] = xi[i, j, :-1].sum() / gamma[i, :-1].sum()

        a /= a.sum(1, keepdims=True)

        for i in range(n_states):
            for j in range(n_obs):
                found = (obs == j).nonzero()[0]
                b[i, j] = gamma[i, found].sum() / gamma[i].sum()

        b /= b.sum(1, keepdims=True)

        pi = gamma[:, 0]

        if np.linalg.norm(a - old_a) < tol and np.linalg.norm(
                b - old_b) < tol or nits > n_iter:
            break

    return pi, a, b


def sample(T, pi, a, b):
    '''Generating observation and states according to pi, a, b
    
    Parameters
    ----------
    T: length of sequence
        int
    pi: start probity   
        (n_states,)
    a: transition matrix  
        (n_statrs, n_states)
    b: emission matrix 
        (n_states, n_obs)

        
    Returns
    -------
    obs: observation sequence [o_1, o_2, ..., o_N]  
        (T, )
    states: hidden states [s_1, s_2, ..., s_N]  
        (T, )        
    '''
    def drawFrom(probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    obs = np.zeros(T, dtype=np.int64)
    states = np.zeros(T, dtype=np.int64)
    states[0] = drawFrom(pi)
    obs[0] = drawFrom(b[int(states[0]), :])

    for t in range(1, T):
        states[t] = drawFrom(a[int(states[t - 1]), :])
        obs[t] = drawFrom(b[int(states[t]), :])

    return obs, states


def test_score():
    model = hmm.MultinomialHMM(n_components=2, init_params='')
    model.startprob_ = init_pi
    model.transmat_ = init_A
    model.emissionprob_ = init_B
    model.n_features = 3

    alpha = HMMfwd(obs, pi=init_pi, a=init_A, b=init_B)
    beta = HMMbwd(obs, a=init_A, b=init_B)
    # my results
    result_1 = alpha[:, -1].sum()
    result_2 = np.sum(init_pi * init_B[:, obs[0]] * beta[:, 0])

    # results from hmmlearn
    result = np.exp(model.score(np.atleast_2d(obs).T))
    assert np.allclose(result_1, result)
    assert np.allclose(result_2, result)


def test_decode():
    model = hmm.MultinomialHMM(n_components=2, init_params='')
    model.startprob_ = init_pi
    model.transmat_ = init_A
    model.emissionprob_ = init_B
    model.n_features = 3

    sequence_1 = Viterbi(obs, pi=init_pi, a=init_A, b=init_B)
    prob, sequence_2 = model.decode(np.atleast_2d(obs).T)
    assert np.allclose(sequence_1, sequence_2)


def test_fit():
    pi, a, b = BaumWelch(
        obs,
        n_states=2,
        n_obs=3,
    )
    print('My results')
    print('pi:\n', pi)
    print('A:\n', a)
    print('B:\n', b)
    print('=' * 50)
    print('results from hmmlearn')
    model = hmm.MultinomialHMM(n_components=2, init_params='ste')
    model.n_features = 3
    model.fit(np.atleast_2d(obs).T)
    print('pi:\n', model.startprob_)
    print('A:\n', model.transmat_)
    print('B:\n', model.emissionprob_)


if __name__ == "__main__":

    # true distribution
    True_pi = np.array([0.5, 0.5])

    True_A = np.array([[0.85, 0.15], [0.12, 0.88]])

    True_B = np.array([[0.8, 0.1, 0.1], [0., 0., 1.]])

    # initial distribution
    init_pi = np.array([0.5, 0.5])

    init_A = np.array([[0.5, 0.5], [0.5, 0.5]])

    init_B = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]])

    obs, states = sample(30, True_pi, True_A, True_B)

    # test
    test_score()
    test_decode()
    test_fit()
