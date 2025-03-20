import jax.numpy as jnp
import diffrax as dfx
import jax
import numpy as np
import jax.experimental.sparse as jesp
import networkx as nx
import optimistix as optx


# Instead of coalescence c we use q which are the coalescent rates
def QQy(t, y, args):
    Q, Q1, q, n= args

    n = q.shape[0]
    # assert Q.shape == (n, n)
    # assert c.shape == (n,)

    yn = y[:-1].reshape(n, n)
    yd = jnp.diag(yn)
    yc = y[-1]
    ret = Q @ yn + yn @ Q.T
    ret -= Q1 * yn
    ret -= jnp.diag(q * yd)
    return jnp.append(ret.reshape(n ** 2), q.dot(yd))

def event_function(tau, t, y, args, **kwargs):
    """Event function to stop when t - tau becomes 0."""
    return tau - t

def f(y0, time_discretization, tau, args):
    Q, Q1, q, n = args
    term = dfx.ODETerm(QQy) 
    solver = dfx.Kvaerno5() 
    saveat = dfx.SaveAt(ts=time_discretization)

    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)  # Root-finding method for event detection
    event = dfx.Event(lambda t, y, args, **kwargs: event_function(tau, t, y, args, **kwargs), root_finder)

    solution = dfx.diffeqsolve(
        term, solver, t0=0, t1=time_discretization[-1], dt0=None, y0=y0, 
        args=args,
        saveat=saveat,
        event = event,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=16)
    )

    return solution

def unit_vector(size, index):
    tmp = jnp.zeros(size)
    tmp = tmp.at[index].set(1)
    return tmp

def solve_ode(time_discretization, init_vertices, q, m, BCOO_indices, index, tau):
    num_nodes = len(q)
    shape = (num_nodes, num_nodes)
    if len(m) == 0:
        copy = jesp.BCOO(((jnp.array([], dtype=jnp.float32), jnp.empty((0, len(shape)), dtype=jnp.int32))), shape=shape)
    else:
        copy = jesp.BCOO((m.astype(float), BCOO_indices.astype(jnp.int64)), shape=shape)
        
    A = copy + copy.transpose()
    e_i = unit_vector(num_nodes, init_vertices[0])
    e_j = unit_vector(num_nodes, init_vertices[1])
    tmp = (e_i.reshape(-1, 1)) @ jnp.array([e_j])
    y0 = jnp.append(jnp.ravel(tmp), 0)

    A1 = A.sum(0).todense()[:, None] + A.sum(1).todense()[None, :]
    args = (A.todense(), A1, q, num_nodes)
    sol = f(y0, time_discretization, tau, args)

    # probabilities = jnp.append(jnp.diff(sol.ys[:,-1]), jnp.array([1 - sol.ys[:,-1][-1]]))
    probabilities = jnp.diff(sol.ys[:,-1])
    prob_not_coal = 1 - sol.ys[:,-1][index]
    return probabilities, sol, prob_not_coal
