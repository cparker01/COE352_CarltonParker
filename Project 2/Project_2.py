import numpy as np
import matplotlib.pyplot as plt

def build_mesh(L=1.,N=11):
    '''
    params:
    L (float) - ending x value
    N (int) - number of nodes

    returns:
    x (np array) - nodal coordinates
    elements (np array) - connectivity
    h (float) - element length (delta x)
    '''

    # defining x, h, and num_elements
    x = np.linspace(0.,L,N)
    h = x[1] - x[0]
    num_elements = N-1

    # connecting elements
    elements = np.zeros((num_elements, 2), dtype=int)
    for e in range(num_elements):
        elements[e][0] = e
        elements[e][1] = e+1

    # return
    return x, elements, h

def element_matrices(h):
    '''
    param:
    h(float) - element length (delta x)

    returns:
    Me (np array) - element mass matrix
    Ke (np array) - element stiffness matrix
    '''
    J = h / 2.
    inv_J = 1. / J

    # 2 point Gauss rule on [-1,1]
    gp = np.array([-1. / np.sqrt(3.), 1. / np.sqrt(3.)])  # Gauss points
    w = np.array([1., 1.])  # weights

    Me = np.zeros((2, 2))
    Ke = np.zeros((2, 2))

    for k in range(2):
        xi = gp[k]

        # linear shape functions on parent element
        N = np.array([0.5 * (1. - xi),
                      0.5 * (1. + xi)])

        dN_dxi = np.array([-0.5, 0.5])

        # mass matrix
        Me += w[k] * np.outer(N, N) * J
        # stiffness matrix
        Ke += w[k] * np.outer(dN_dxi, dN_dxi) * inv_J

    return Me, Ke

def assemble(x, elements, Me, Ke):
    '''
    This function assembles the mass and stiffness matrices.

    params:
    x (np array) - nodal coordinates
    elements (np array) - connectivity
    Me (np array) - element mass matrix
    Ke (np array) - element stiffness matrix

    returns:
    M (np array) - global mass matrix
    K (np array) - global stiffness matrix
    '''

    N = len(x) # number of nodes
    num_elements = elements.shape[0]

    M = np.zeros((N, N))
    K = np.zeros((N, N))

    for e in range(num_elements):
        n1, n2 = elements[e]

        # add element mass
        M[n1, n1] += Me[0, 0]
        M[n1, n2] += Me[0, 1]
        M[n2, n1] += Me[1, 0]
        M[n2, n2] += Me[1, 1]

        # add element stiffness
        K[n1, n1] += Ke[0, 0]
        K[n1, n2] += Ke[0, 1]
        K[n2, n1] += Ke[1, 0]
        K[n2, n2] += Ke[1, 1]

    return M, K

def f_source(x, t):
    return (np.pi**2 - 1.) * np.exp(-t) * np.sin(np.pi * x)

def assemble_f(x, elements, t, f_handle):
    '''
    This function assembles the global load vector.

    params:
    x (np array) - nodal coordinates
    elements (np array) - connectivity
    t (float) - time
    f_handle (callable) - source function

    returns:
    F (np array) - global load vector at t
    '''

    N = len(x)
    num_elements = elements.shape[0]

    F = np.zeros(N)

    # 2 point Guass rule on [-1, 1]
    gp = np.array([-1. / np.sqrt(3.), 1. / np.sqrt(3.)]) # Guass points
    w = np.array([1., 1.]) # weights


    for e in range(num_elements):
        n1, n2 = elements[e]
        x1, x2 = x[n1], x[n2]

        Fe = np.zeros(2) # local load vector

        x_mid = (x1 + x2) / 2
        half_h = (x2 - x1) / 2

        for k in range(2):
            # linear shape functions
            xi = gp[k]
            Nk = np.array([(1. - xi) / 2,  # N1
                           (1. + xi) / 2])  # N2

            # map Gauss point to x
            xk = x_mid + half_h * xi
            fk = f_handle(xk, t)

            Fe += w[k] * fk * Nk * half_h

        # assemble into global F
        F[n1] += Fe[0]
        F[n2] += Fe[1]

    return F

def apply_dirichlet(M, K, bound_nodes):
    '''
    params:
    M (np array) - global mass matrix
    K (np array) - global stiffness matrix
    F (np array) - global load vector
    bound_nodes - indices of Dirichlet boundary nodes

    returns:
    Mi (np array) - mass matrix for interior nodes
    Ki (np array) - stiffness matrix for interior nodes
    interior_nodes (np array) - indices of interior nodes
    '''


    N = M.shape[0]
    all_nodes = np.arange(N)
    interior_nodes = np.setdiff1d(all_nodes, bound_nodes)

    Mi = M[np.ix_(interior_nodes, interior_nodes)]
    Ki = K[np.ix_(interior_nodes, interior_nodes)]

    return Mi, Ki, interior_nodes

def u_initial(x):
    return np.sin(np.pi * x)

def forward_euler(Mi, Ki, x, elements, interior_nodes, U0, dt, T_fin, f_handle):
    '''
    params:
    Mi (np array) - mass matrix for interior nodes
    Ki (np array) - stiffness matrix for interior nodes
    x (np array) - nodal coordinates
    elements (np array) - connectivity
    interior_nodes (np array) - indices of interior nodes
    U0 (np array) - initial interior solution
    dt (float) - time step size
    T_fin (float) - final time
    f_handle (callable) - source term

    returns:
    times (np array) - time values
    U_history (np array) - interior solutions at each time step
    '''


    Ni = len(U0)
    Nt = int(np.round(T_fin/dt))

    U = U0.copy()
    U_history = np.zeros((Nt+1, Ni))
    times = np.zeros(Nt + 1)
    U_history[0, :] = U
    times[0] = 0.
    t = 0.

    for n in range(Nt):
        # assemble global F at time t
        F_full = assemble_f(x, elements, t, f_handle)
        Fi = F_full[interior_nodes]


        # FE
        rhs = Mi @ U + dt * (Fi - Ki @ U)
        U = np.linalg.solve(Mi, rhs)

        t += dt
        times[n + 1] = t
        U_history[n + 1, :] = U


    return times, U_history

def reconstruct_full(U_int, interior_nodes, N):
    '''
    This function reconstructs full solution vector.

    params:
    U_int (np array) - interior solution
    interior_nodes (np array) - indices of interior nodes
    N (int) - number of nodes

    return:
    U_full (np array) - solution including boundaries
    '''

    U_full = np.zeros(N)
    U_full[interior_nodes] = U_int
    return U_full

def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

def backward_euler(Mi, Ki, x, elements, interior_nodes,
                   U0, dt, T_fin, f_handle):
    '''
    params:
    Mi (np array) - interior mass matrix
    Ki (np array) - interior stiffness matrix
    x (np array) - nodal coordinates
    elements (np array) - connectivity
    interior_nodes (np array) - indices of interior nodes
    U0 (np array) - initial interior solution
    dt (float) - time step size
    T_fin (float) - final time
    f_handle (callable) - source term

    returns:
    times (np array) - time values
    U_history (np array) - interior solutions at each time step
    '''


    Ni = len(U0)
    Nt = int(np.round(T_fin / dt))

    U = U0.copy()
    U_history = np.zeros((Nt+1, Ni))
    times = np.zeros(Nt + 1)

    U_history[0, :] = U
    times[0] = 0.
    t = 0.


    A = Mi + dt * Ki
    for n in range(Nt):
        t_next = dt+t

        # assemble F
        F_full_next = assemble_f(x, elements, t_next, f_handle)
        Fi_next = F_full_next[interior_nodes]



        # BE
        rhs = Mi @ U+dt * Fi_next
        U = np.linalg.solve(A, rhs)

        t = t_next
        times[n+1] = t
        U_history[n+1, :] = U

    return times, U_history


if __name__ == '__main__':

    # declaring variables
    L = 1.
    N = 11
    dt = 1.
    t0 = 0.
    T_fin = 1.

    # build mesh and global matrices
    x, elements, h = build_mesh(L, N)
    Me, Ke = element_matrices(h)
    M, K = assemble(x, elements, Me, Ke)

    # load vector at t=0
    F0 = assemble_f(x, elements, t0, f_source)
    # apply Dirichlet boundary conditions
    boundary_nodes = [0, N-1]
    Mi, Ki, interior_nodes = apply_dirichlet(M, K, boundary_nodes)

    # initial conditions of interior nodes
    U0_full = u_initial(x)
    U0 = U0_full[interior_nodes]

    # FE and BE time stepping
    times_FE, Uhistory_FE = forward_euler(Mi, Ki, x, elements, interior_nodes,
                                          U0, dt, T_fin, f_source)

    times_BE, Uhistory_BE = backward_euler(Mi, Ki, x, elements, interior_nodes,
                                           U0, dt, T_fin, f_source)

    # reconstruct full solution at final time
    N_nodes = len(x)
    U_FE_int_final = Uhistory_FE[-1, :]
    U_BE_int_final = Uhistory_BE[-1, :]

    U_FE_full = reconstruct_full(U_FE_int_final, interior_nodes, N_nodes)
    U_BE_full = reconstruct_full(U_BE_int_final, interior_nodes, N_nodes)

    t_final = times_FE[-1]
    u_ex = u_exact(x, t_final)

    err_FE = U_FE_full - u_ex
    err_BE = U_BE_full - u_ex

    # printing max error for FE and BE
    print("max error for FE:", np.max(np.abs(err_FE)))
    print("max error for BE:", np.max(np.abs(err_BE)))

    # plotting results
    plt.plot(x, u_ex, label="Exact")
    plt.plot(x, U_FE_full, 'o--', label="Forward Euler")
    plt.plot(x, U_BE_full, 's--', label="Backward Euler")
    plt.xlabel("x")
    plt.ylabel("u(x, t=1)")
    plt.legend()
    plt.grid(True)
    plt.show()



