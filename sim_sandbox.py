import numpy as np


class Point:
    def __init__(self, x, v, m):
        self.pos = np.array(x, dtype=float)
        self.vel = np.array(v, dtype=float)
        self.m = m

    def __str__(self):
        return f"Pos {self.pos} --- vel {self.vel} --- mass {self.m}"


# Octree implementation WIP. For Barnes-Hut algorithm (and FMM), when calculating
# per particle what force is exerted upon it by the rest of the system, instead of
# calculating the force from each particle, we can group particles that are far
# away from the particle (some parameter) into a single particle with combined mass. etc etc wip wip.
class Node:
    def __init__(self, octree, parent, offset, length):
        self.octree = octree
        self.parent = parent
        self.offset = offset
        self.length = length
        # Each element representing a child node aka the parent node of the
        # octree in that octant of this node.
        self.octs = [None, None, None, None, None, None, None, None]

    def new_offset(oct_idx, oct_len):
        return np.array([oct_idx & 1,
                        (oct_idx >> 1) & 1,
                        (oct_idx >> 2) & 1]) * oct_len

    def make_oct(self, oct_idx):
        self.octs[oct_idx] = Node(self.octree,
                                  self,
                                  self.new_offset(oct_idx, self.length),
                                  self.length / 2)

    def make_all_oct(self):
        for i in range(8):
            self.make_oct(i)

    # When building the tree, we need to insert a point into the tree. Either this function, or some
    # external function will recursively build the whole tree given a list of points, root node and our
    # grouping condition / distance threshold etc etc wip wip.
    def insert(self, point):
        pass


# simple n body shit: ---------------------------------------------------------

def force(A, B):
    # G = 6.67408e-11
    G = 1

    # Force on point A due to point B
    F = G * A.m * B.m / np.linalg.norm(B.pos - A.pos)**3 * (B.pos - A.pos)
    return F


def accel(A, B):
    return force(A, B) / A.m


# N body with gauss-legendre
def integrated(dt, points=None, steps=3):
    if points is None:
        points = [Point([100, 100, 100], [0, 0, 0], 100),
                  Point([800, 800, 800], [5, 5, 5], 10)]
    nodes, weights = np.polynomial.legendre.leggauss(steps)

    # Scale nodes to the time interval [0, dt]
    t_nodes = 0.5 * dt * (nodes + 1)

    # new_positions = [np.copy(p.pos) for p in points]
    # new_velocities = [np.copy(p.vel) for p in points]

    for k in range(10):
        print(f"At step {k} ----------------")
        print(points[0])
        print(points[1])

        for i, p1 in enumerate(points):
            total_acc = np.zeros(3)

            for j, t in enumerate(t_nodes):
                acc_at_t = np.zeros(3)
                # Compute acceleration due to all other points at quadrature nodes

                for p2 in points:
                    if p1 != p2:
                        acc_at_t += accel(p1, p2)

                total_acc += weights[j] * acc_at_t

            p1.vel += total_acc * dt
            p1.pos += p1.vel * dt

            # Update velocity and position using the weighted acceleration
        #     new_velocities[i] += total_acc * dt
        #     new_positions[i] += new_velocities[i] * dt

        # # Apply updates
        # for i, p in enumerate(points):
        #     p.pos = new_positions[i]
        #     p.vel = new_velocities[i]

    return points


def two_body():
    p1 = Point([100, 100, 100], [0, 0, 0], 100)
    p2 = Point([800, 800, 800], [5, 5, 5], 10)

    dt = 1

    for i in range(10):
        print(f"At step {i} ----------------")
        print("P1", p1)
        print("P2", p2)

        a1 = accel(p1, p2)
        a2 = accel(p2, p1)

        p1.vel += a1 * dt
        p2.vel += a2 * dt

        p1.pos += p1.vel * dt
        p2.pos += p2.vel * dt

    return [p1, p2]


def n_body(points=None):
    if points is None:
        points = [Point([100, 100, 100], [0, 0, 0], 100),
                  Point([800, 800, 800], [5, 5, 5], 10)]

    dt = 1

    for i in range(10):
        for p, pi in enumerate(points):
            print(f"At step {i} ----------------")
            print(f"P{p}", pi)

        for p1 in points:
            for p2 in points:
                if p1 != p2:
                    a1 = accel(p1, p2)
                    p1.vel += a1 * dt

        for p1 in points:
            p1.pos += p1.vel * dt

    return points


if __name__ == "__main__":
    tbp = two_body()
    nbp = n_body()
    np.testing.assert_allclose(tbp[0].pos, nbp[0].pos)
    np.testing.assert_allclose(tbp[0].vel, nbp[0].vel)

    gbp = integrated(1, None, 3)
