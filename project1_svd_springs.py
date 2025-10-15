from svd_decompose import svd_decompose, condition_number, svd_inverse
import numpy as np


def build_stiffness_matrix(k):
    """
    This function builds the stiffness matrix K.

    Parameters:
        k - array : spring constants

    """

    n = len(k) - 1 # number of masses
    K = np.zeros((n, n)) # initializing K

    # filling K matrix
    for i in range(n):
        # main diagonal
        K[i, i] = k[i] + k[i + 1]

        # off diagonals
        if i > 0:
            K[i, i - 1] = -k[i]
        if i < n - 1:
            K[i, i + 1] = -k[i + 1]
    return K

def solve_with_svd(K, F):
    """
    This function solves the equation Ku=F using the inverse matrix. If
    the matrix K is singluar, it prints a warning and returns none.

    Parameters:
        K - np array : stiffness matrix
        F - np array : external force vector

    """

    try:
        # computing the inverse of K
        Kinv = svd_inverse(K)
        # computing displacements
        u = Kinv @ F

        return u
    except ValueError:
        print("Matrix is singular.")
        return None

def elongations_and_stresses(u, k):
    """
    This function calculates the elongation and internal stressed of each spring.

    parameters:
    u - array : displacement vector
    k - array : spring constants


    """

    n = len(k) - 1 # number of masses

    # initializing arrays
    elong = np.zeros(n + 1)
    stress = np.zeros(n + 1)

    # handling boundary conditions
    left_disp = 0.0 if k[0] != 0 else u[0]
    right_disp = 0.0 if k[-1] != 0 else u[-1]

    displacements = np.concatenate(([left_disp], u, [right_disp]))

    # computing elongation and stress for each spring
    for i in range(n + 1):
        elong[i] = displacements[i + 1] - displacements[i]
        stress[i] = k[i] * elong[i]

    return elong, stress

if __name__ == "__main__":
    print("Spring–Mass System solver\n")
    n = int(input("How many masses are in your system? "))

    # getting spring constants
    print(f"Enter {n + 1} spring constants separated by spaces (k0...k{n}):")
    k = np.array(list(map(float, input().split())))

    # getting boundary conditions
    left_fixed = input("Left end fixed? (y/n): ").lower() == 'y'
    right_fixed = input("Right end fixed? (y/n): ").lower() == 'y'

    # adjust k values for boundary conditions
    if not left_fixed:
        k[0] = 0.0 # free left side
    if not right_fixed:
        k[-1] = 0.0 # free right side

    # optional masses (not used)
    print(f"Enter {n} masses separated by spaces (press Enter to skip):")
    m_input = input().strip()
    if m_input:
        masses = np.array(list(map(float, m_input.split())))
    else:
        masses = np.ones(n)

    # external forces
    F = np.zeros(n)
    F[0] = 1.0

    # print statements

    K = build_stiffness_matrix(k)
    print("K =\n", K)

    U, S, Vt = svd_decompose(K)

    print("\nSingular values :", S)
    print("Condition number :", condition_number(S))
    print("Eigenvalues of K :", np.sort(np.linalg.eigvalsh(K)))

    u = solve_with_svd(K, F)

    print("\nDisplacements (u) =", u)

    elongations, stress = elongations_and_stresses(u, k)
    print("\nElongations =", elongations)
    print("Stresses =", stress)

    if len(S) == 0 or S[-1] <= 1e-6:
        print("Note : K is rank deficient.")



