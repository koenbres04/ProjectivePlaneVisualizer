import numpy as np
import typing


EPSILON = 1e-10


def swap_row(A, i, j):
    temp = A[i, :].copy()
    A[i, :] = A[j, :]
    A[j, :] = temp


def scale_row(A, i, s):
    A[i, :] *= s


def add_scaled_row(A, i, j, s):
    A[i, :] += s*A[j, :]


def get_reduced(A: np.ndarray) -> np.ndarray:
    # make a copy of the matrix A
    A = A.copy()
    # index of the current row
    n = 0
    # do the reduced echelon algorithm
    for j in range(A.shape[1]):
        for i in range(n, A.shape[0]):
            if abs(A[i, j]) >= EPSILON:
                if i != n:
                    swap_row(A, n, i)
                break
        else:
            continue
        scale_row(A, n, 1/A[n, j])
        for k in range(A.shape[0]):
            if k != n:
                add_scaled_row(A, k, n, -A[k, j])
        n += 1
    # remove lower rows that consist only of zeros
    i = A.shape[0]-1
    while i >= 0:
        found_nonzero = False
        for j in range(A.shape[1]):
            if abs(A[i, j]) >= EPSILON:
                found_nonzero = True
                break
        if found_nonzero:
            break
        i -= 1
    return A[:i+1, :]


class Subspace:
    def __init__(self, reduced_matrix: np.ndarray):
        self.A = reduced_matrix

    @classmethod
    def kernel(cls, matrix: np.ndarray):
        return cls(get_reduced(matrix))

    @classmethod
    def perp_to(cls, *vectors):
        matrix = np.array(vectors)
        return cls.kernel(matrix)

    @classmethod
    def from_generators(cls, *generators):
        return cls.perp_to(*generators).perp

    @classmethod
    def intersection(cls, a, b):
        return cls.kernel(np.concatenate((a.A, b.A)))

    @property
    def pivots(self):
        pivots = []
        last_k = -1
        for j in range(self.A.shape[1]):
            k = -1
            for i in range(self.A.shape[0]):
                if k == -1 and abs(self.A[i, j]-1) < EPSILON:
                    k = i
                elif abs(self.A[i, j]) > EPSILON:
                    continue
            if k != -1 and k > last_k:
                last_k = k
                pivots.append((k, j))
        return pivots

    @property
    def dimension(self):
        return self.A.shape[0]

    def get_generators(self) -> typing.List[np.ndarray]:
        generators = []
        pivots = self.pivots
        pivot_js = [j for i, j in pivots]
        not_pivot_js = [j for j in range(self.A.shape[1]) if j not in pivot_js]
        for k in not_pivot_js:
            w = np.zeros(self.A.shape[1])

            for i in range(self.A.shape[0]):
                w[pivots[i][1]] = -self.A[i, k]
            w[k] = 1
            generators.append(w)
        return generators

    @property
    def perp(self):
        return self.perp_to(*self.get_generators())

    def __add__(self, other):
        return self.intersection(self.perp, other.perp).perp

    def transform(self, inv_transformation):
        return Subspace.kernel(self.A.dot(inv_transformation))
