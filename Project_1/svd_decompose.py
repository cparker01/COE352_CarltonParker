import numpy as np

def svd_decompose(A):
    # forming AtA
    AtA = A.T@A

    # computing eigenvalues
    evals, V = np.linalg.eigh(AtA)

    # sorting values in descending order
    evals = evals[::-1]
    V = V[:,::-1]

    # calculating singular values
    S = []
    for val in evals:
        if val > 0:
            S.append(np.sqrt(val))
        else:
            S.append(0.0)
    S = np.array(S)

    # initializing U matrix
    n, m = A.shape
    U = np.zeros((n, len(S)))

    # computing columns of U
    for i in range(len(S)):
        sigma = S[i]
        if sigma > 1e-12:
            vi = V[:, i]
            ui = A @ vi
            ui = ui / sigma
            U[:, i] = ui
        else:
            U[:, i] = 0.0

    # t transpose
    Vt = V.T

    return U, S, Vt

# calculating and returning condition number
def condition_number(S):
    s_max = np.max(S)
    s_min = np.min(S)

    return s_max / s_min if s_min > 0 else np.inf

# calculating inverse
def svd_inverse(A):
    U, S, Vt = svd_decompose(A)

    # testing for singular matrix
    if np.any(S < 1e-12):
        raise ValueError("Matrix is singular.")
    Sinv = np.diag(1 / S)
    return Vt.T @ Sinv @ U.T

# testing svd_decompose function
if __name__ == "__main__":
    A = np.array([[3., 1.], [2., 4.]])
    U, S, Vt = svd_decompose(A)
    print("U =\n", U)
    print("S =", S)
    print("Vt =\n", Vt)

    # Condition number
    cond_num = condition_number(S)
    print("\nCondition number =", cond_num)

    # testing svd with pythons built-in function
    U_np, S_np, Vt_np = np.linalg.svd(A)
    print("\nbuilt in numpy SVD :")
    print("Singular values =", S_np)

    # getting inverse of A using my svd function
    try:
        Ainv = svd_inverse(A)
        print("\nInverse of A using my SVD :")
        print(Ainv)

        # checking if the inverse works with A * A⁻¹ identity
        print("\nA multiplied by its inverse :")
        print(A @ Ainv)

    # if matrix can't be inverted, show a message
    except ValueError:
        print("\nMatrix is singular.")

    # inverse if full rank
    if np.all(S > 1e-12):
        Ainv = Vt.T @ np.diag(1 / S) @ U.T
        print("\nA⁻¹ (SVD) =\n", Ainv)
        print("A @ A⁻¹ ≈ I:\n", A @ Ainv)
    else:
        print("Matrix is singular.")