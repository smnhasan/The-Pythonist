# Competitive Programming DSA Reference (Python)

A collection of clean, ready-to-use implementations for common mid-level competitive programming algorithms.

---

## Table of Contents

1. [Binary Search (including "Search on Answer" / Parametric Search)](#1-binary-search-including-search-on-answer-parametric-search)
2. [Classic DP — 0/1 Knapsack, Unbounded Knapsack, LCS, LIS (O(n log n))](#2-classic-dp-01-knapsack-unbounded-knapsack-lcs-lis-on-log-n)
3. [BFS / DFS (including Multi-Source BFS)](#3-bfs-dfs-including-multi-source-bfs)
4. [Dijkstra's Algorithm](#4-dijkstras-algorithm)
5. [Bellman-Ford (Negative Weights, Cycle Detection)](#5-bellman-ford-negative-weights-cycle-detection)
6. [Union-Find (Disjoint Set Union) with Path Compression + Union by Rank](#6-union-find-disjoint-set-union-with-path-compression-union-by-rank)
7. [Segment Tree (Range Sum / Min / Max, with Lazy Propagation)](#7-segment-tree-range-sum-min-max-with-lazy-propagation)
8. [Sieve of Eratosthenes & Segmented Sieve](#8-sieve-of-eratosthenes-segmented-sieve)
9. [Modular Exponentiation](#9-modular-exponentiation)
10. [Bitmasking Techniques](#10-bitmasking-techniques)
11. [XOR Properties](#11-xor-properties)
12. [Counting Set Bits & Subset Enumeration via Bitmask](#12-counting-set-bits-subset-enumeration-via-bitmask)

---

## 1. Binary Search (including "Search on Answer" / Parametric Search)

```python
def binary_search(arr, target):
    """Standard binary search. Returns index of target, or -1 if not found."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr, target):
    """First index where arr[index] >= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(arr, target):
    """First index where arr[index] > target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search_on_answer(lo, hi, feasible):
    """
    Generic 'binary search on answer' / parametric search.
    Finds the smallest x in [lo, hi] for which feasible(x) is True,
    assuming feasible() is monotonic: False False False True True True.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# Example: minimum capacity to ship packages within `days` days
def min_capacity_to_ship(weights, days):
    def feasible(capacity):
        needed_days, current_load = 1, 0
        for w in weights:
            if current_load + w > capacity:
                needed_days += 1
                current_load = 0
            current_load += w
        return needed_days <= days

    lo, hi = max(weights), sum(weights)
    return binary_search_on_answer(lo, hi, feasible)


if __name__ == "__main__":
    arr = [1, 3, 3, 5, 7, 9, 9, 11]
    print(binary_search(arr, 7))          # 4
    print(lower_bound(arr, 3))            # 1
    print(upper_bound(arr, 3))            # 3
    print(min_capacity_to_ship([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  # 15
```

---

## 2. Classic DP — 0/1 Knapsack, Unbounded Knapsack, LCS, LIS (O(n log n))

```python
def knapsack_01(weights, values, capacity):
    """0/1 Knapsack: each item used at most once. Returns max value."""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        # iterate capacity backwards so each item is used at most once
        for c in range(capacity, weights[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]


def knapsack_unbounded(weights, values, capacity):
    """Unbounded Knapsack: unlimited copies of each item allowed."""
    dp = [0] * (capacity + 1)
    for c in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= c:
                dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]


def longest_common_subsequence(s1, s2):
    """Returns length of LCS between s1 and s2."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def longest_increasing_subsequence(arr):
    """O(n log n) LIS length using binary search (patience sorting)."""
    import bisect
    tails = []  # tails[i] = smallest tail value of an increasing subsequence of length i+1
    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


if __name__ == "__main__":
    print(knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7))   # 9
    print(knapsack_unbounded([2, 3, 5], [3, 4, 6], 8))  # 12 (four items of weight 2, value 3 each)
    print(longest_common_subsequence("ABCBDAB", "BDCABA"))  # 4
    print(longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
```

---

## 3. BFS / DFS (including Multi-Source BFS)

```python
from collections import deque

def bfs(graph, start):
    """Standard BFS. graph: dict[node] -> list of neighbors."""
    visited = {start}
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs_iterative(graph, start):
    """Iterative DFS using an explicit stack."""
    visited = {start}
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return order


def dfs_recursive(graph, node, visited=None, order=None):
    """Recursive DFS."""
    if visited is None:
        visited, order = set(), []
    visited.add(node)
    order.append(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, order)
    return order


def multi_source_bfs(graph, sources):
    """
    BFS starting from multiple sources simultaneously.
    Returns dist: dict[node] -> shortest distance from the nearest source.
    Useful for problems like 'distance to nearest 1 in a grid'.
    """
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


if __name__ == "__main__":
    g = {
        1: [2, 3],
        2: [1, 4],
        3: [1, 4],
        4: [2, 3, 5],
        5: [4],
    }
    print(bfs(g, 1))                       # [1, 2, 3, 4, 5]
    print(dfs_iterative(g, 1))             # [1, 3, 4, 5, 2]
    print(dfs_recursive(g, 1))             # [1, 2, 4, 3, 5]
    print(multi_source_bfs(g, [1, 5]))     # {1: 0, 5: 0, 2: 1, 3: 1, 4: 1}
```

---

## 4. Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    """
    Single-source shortest path with non-negative weights.
    graph: dict[node] -> list of (neighbor, weight)
    Returns dist: dict[node] -> shortest distance from start.
    """
    dist = {start: 0}
    pq = [(0, start)]
    visited = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in graph.get(node, []):
            new_dist = d + weight
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist


if __name__ == "__main__":
    g = {
        'A': [('B', 4), ('C', 1)],
        'B': [('A', 4), ('C', 2), ('D', 5)],
        'C': [('A', 1), ('B', 2), ('D', 8)],
        'D': [('B', 5), ('C', 8)],
    }
    print(dijkstra(g, 'A'))  # {'A': 0, 'C': 1, 'B': 3, 'D': 8}
```

---

## 5. Bellman-Ford (Negative Weights, Cycle Detection)

```python
def bellman_ford(num_nodes, edges, start):
    """
    edges: list of (u, v, weight) — works with negative weights.
    Returns (dist, has_negative_cycle).
    dist[i] = shortest distance from start to node i (float('inf') if unreachable).
    """
    dist = [float('inf')] * num_nodes
    dist[start] = 0

    # Relax all edges (V - 1) times
    for _ in range(num_nodes - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # One more pass to detect negative cycles
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle


if __name__ == "__main__":
    # nodes 0..4
    edges = [
        (0, 1, 4), (0, 2, 5),
        (1, 2, -3), (2, 3, 4),
        (3, 1, -6), (1, 4, 5),
    ]
    dist, neg_cycle = bellman_ford(5, edges, 0)
    print(dist, neg_cycle)  # has negative cycle through 1->2->3->1
```

---

## 6. Union-Find (Disjoint Set Union) with Path Compression + Union by Rank

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # number of disjoint components

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # already in the same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


if __name__ == "__main__":
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    print(uf.connected(0, 2))  # True
    print(uf.connected(0, 3))  # False
    print(uf.count)            # 3 components: {0,1,2}, {3,4}, {5}
```

---

## 7. Segment Tree (Range Sum / Min / Max, with Lazy Propagation)

```python
class SegmentTree:
    """
    Generic segment tree supporting range queries (sum/min/max) and
    range updates with lazy propagation (range add).
    Change `merge` and the identity element to switch between sum/min/max.
    """
    def __init__(self, arr, merge=lambda a, b: a + b, identity=0):
        self.n = len(arr)
        self.merge = merge
        self.identity = identity
        self.tree = [identity] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
            return
        mid = (l + r) // 2
        self._build(arr, 2 * node, l, mid)
        self._build(arr, 2 * node + 1, mid + 1, r)
        self.tree[node] = self.merge(self.tree[2 * node], self.tree[2 * node + 1])

    def _push_down(self, node, l, r):
        if self.lazy[node] != 0:
            mid = (l + r) // 2
            for child, cl, cr in ((2 * node, l, mid), (2 * node + 1, mid + 1, r)):
                self.tree[child] += self.lazy[node] * (cr - cl + 1)
                self.lazy[child] += self.lazy[node]
            self.lazy[node] = 0

    def update_range(self, ql, qr, val):
        """Add `val` to every element in range [ql, qr]."""
        self._update_range(1, 0, self.n - 1, ql, qr, val)

    def _update_range(self, node, l, r, ql, qr, val):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.tree[node] += val * (r - l + 1)
            self.lazy[node] += val
            return
        self._push_down(node, l, r)
        mid = (l + r) // 2
        self._update_range(2 * node, l, mid, ql, qr, val)
        self._update_range(2 * node + 1, mid + 1, r, ql, qr, val)
        self.tree[node] = self.merge(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        """Query the merge (sum/min/max) over range [ql, qr]."""
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return self.identity
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push_down(node, l, r)
        mid = (l + r) // 2
        left = self._query(2 * node, l, mid, ql, qr)
        right = self._query(2 * node + 1, mid + 1, r, ql, qr)
        return self.merge(left, right)


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11]

    # Range Sum Segment Tree
    seg_sum = SegmentTree(arr, merge=lambda a, b: a + b, identity=0)
    print(seg_sum.query(1, 3))      # 3+5+7 = 15
    seg_sum.update_range(1, 3, 10) # add 10 to indices 1..3
    print(seg_sum.query(1, 3))      # 15 + 30 = 45
    print(seg_sum.query(0, 5))      # full sum after update

    # Range Min Segment Tree (note: lazy propagation here only supports "add",
    # which is valid for min/max merges too — adding shifts min/max correctly)
    seg_min = SegmentTree(arr, merge=min, identity=float('inf'))
    print(seg_min.query(0, 5))      # 1
    print(seg_min.query(2, 4))      # 5
```

---

## 8. Sieve of Eratosthenes & Segmented Sieve

```python
def sieve_of_eratosthenes(n):
    """Returns a boolean list is_prime[0..n] and list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return is_prime, primes


def segmented_sieve(low, high):
    """
    Finds all primes in range [low, high] using a segmented sieve.
    Efficient when high is large but (high - low) is small.
    """
    if high < 2:
        return []

    limit = int(high ** 0.5) + 1
    _, base_primes = sieve_of_eratosthenes(limit)

    is_prime = [True] * (high - low + 1)
    if low == 0:
        is_prime[0] = False
    if low <= 1 <= high:
        is_prime[1 - low] = False

    for p in base_primes:
        # Start marking from the first multiple of p that is >= low
        start = max(p * p, ((low + p - 1) // p) * p)
        for multiple in range(start, high + 1, p):
            is_prime[multiple - low] = False

    return [low + i for i, prime in enumerate(is_prime) if prime]


if __name__ == "__main__":
    is_prime, primes = sieve_of_eratosthenes(50)
    print(primes)  # [2, 3, 5, 7, 11, ..., 47]

    print(segmented_sieve(90, 130))  # primes between 90 and 130
```

---

## 9. Modular Exponentiation

```python
def mod_pow(base, exponent, mod):
    """Computes (base^exponent) % mod efficiently using binary exponentiation."""
    result = 1
    base %= mod
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exponent >>= 1
    return result


def mod_inverse(a, mod):
    """
    Modular inverse of a under prime modulus `mod`, via Fermat's Little Theorem.
    Requires mod to be prime and gcd(a, mod) == 1.
    """
    return mod_pow(a, mod - 2, mod)


if __name__ == "__main__":
    print(mod_pow(2, 10, 1000000007))    # 1024
    print(mod_pow(7, 128, 13))           # fast modular power
    print(mod_inverse(3, 1000000007))    # inverse of 3 mod 1e9+7
    # sanity check: (3 * inverse) % mod should be 1
    print((3 * mod_inverse(3, 1000000007)) % 1000000007)  # 1
```

---

## 10. Bitmasking Techniques

```python
def is_bit_set(num, i):
    """Check if bit i (0-indexed from LSB) is set in num."""
    return (num >> i) & 1 == 1


def set_bit(num, i):
    """Set bit i in num."""
    return num | (1 << i)


def clear_bit(num, i):
    """Clear (unset) bit i in num."""
    return num & ~(1 << i)


def toggle_bit(num, i):
    """Toggle bit i in num."""
    return num ^ (1 << i)


def lowest_set_bit(num):
    """Returns the value of the lowest set bit (isolates it)."""
    return num & (-num)


def all_subsets_bitmask(n):
    """
    Generate all 2^n subsets of a set of size n as bitmasks.
    Each bitmask represents which elements (0..n-1) are included.
    """
    subsets = []
    for mask in range(1 << n):
        subset = [i for i in range(n) if is_bit_set(mask, i)]
        subsets.append(subset)
    return subsets


def submask_enumeration(mask):
    """
    Enumerate all submasks of a given bitmask (including 0 and mask itself).
    Classic O(3^n) technique when done over all masks.
    """
    submasks = []
    sub = mask
    while True:
        submasks.append(sub)
        if sub == 0:
            break
        sub = (sub - 1) & mask
    return submasks


if __name__ == "__main__":
    num = 0b1010  # 10

    print(is_bit_set(num, 1))    # True (bit 1 is set)
    print(is_bit_set(num, 0))    # False
    print(bin(set_bit(num, 0)))  # 0b1011
    print(bin(clear_bit(num, 1))) # 0b1000
    print(bin(toggle_bit(num, 2))) # 0b1110
    print(lowest_set_bit(12))    # 4  (12 = 1100, lowest set bit = 0100)

    print(all_subsets_bitmask(3))
    # [[], [0], [1], [0,1], [2], [0,2], [1,2], [0,1,2]]

    print(submask_enumeration(0b101))  # [5, 4, 1, 0]
```

---

## 11. XOR Properties

```python
def find_single_number(arr):
    """
    Given an array where every element appears twice except one,
    find that single element. Uses XOR property: a^a = 0, a^0 = a.
    """
    result = 0
    for num in arr:
        result ^= num
    return result


def find_two_unique_numbers(arr):
    """
    Given an array where every element appears twice except exactly two,
    find those two unique elements in O(n) time, O(1) space.
    """
    xor_all = 0
    for num in arr:
        xor_all ^= num

    # Isolate a bit that differs between the two unique numbers
    diff_bit = xor_all & (-xor_all)

    num1 = num2 = 0
    for num in arr:
        if num & diff_bit:
            num1 ^= num
        else:
            num2 ^= num
    return num1, num2


def max_xor_pair_bruteforce(arr):
    """Naive O(n^2) max XOR pair — baseline for comparison with Trie approach."""
    best = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            best = max(best, arr[i] ^ arr[j])
    return best


def xor_range(n):
    """
    Computes XOR of all numbers from 0 to n in O(1) using the pattern:
    XOR(0..n) cycles every 4 numbers: n, 1, n+1, 0
    """
    mod = n % 4
    if mod == 0:
        return n
    elif mod == 1:
        return 1
    elif mod == 2:
        return n + 1
    else:
        return 0


if __name__ == "__main__":
    print(find_single_number([4, 1, 2, 1, 2]))         # 4
    print(find_two_unique_numbers([1, 2, 1, 3, 2, 5]))  # (3, 5) order may vary
    print(max_xor_pair_bruteforce([3, 10, 5, 25, 2, 8]))  # 28 (5^25)
    print(xor_range(5))  # XOR of 0,1,2,3,4,5 = 1
```

---

## 12. Counting Set Bits & Subset Enumeration via Bitmask

```python
def count_set_bits_naive(num):
    """O(log n) bit counting by checking each bit."""
    count = 0
    while num:
        count += num & 1
        num >>= 1
    return count


def count_set_bits_brian_kernighan(num):
    """
    Brian Kernighan's algorithm: O(number of set bits).
    Each iteration clears the lowest set bit.
    """
    count = 0
    while num:
        num &= num - 1  # clears the lowest set bit
        count += 1
    return count


def count_set_bits_builtin(num):
    """Using Python's built-in (fast, implemented in C)."""
    return bin(num).count('1')  # or num.bit_count() in Python 3.10+


def precompute_set_bits(n):
    """
    DP-based precomputation of set bit counts for all numbers 0..n.
    dp[i] = dp[i >> 1] + (i & 1)
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


def enumerate_subsets_with_sum(arr):
    """
    Enumerate all subsets of arr using bitmask, computing sum for each.
    Returns list of (subset_elements, subset_sum).
    """
    n = len(arr)
    results = []
    for mask in range(1 << n):
        subset = []
        total = 0
        for i in range(n):
            if mask & (1 << i):
                subset.append(arr[i])
                total += arr[i]
        results.append((subset, total))
    return results


if __name__ == "__main__":
    print(count_set_bits_naive(13))               # 1101 -> 3
    print(count_set_bits_brian_kernighan(13))      # 3
    print(count_set_bits_builtin(13))              # 3
    print(precompute_set_bits(10))                 # [0,1,1,2,1,2,2,3,1,2,2]

    print(enumerate_subsets_with_sum([1, 2, 3]))
    # [([], 0), ([1], 1), ([2], 2), ([1,2], 3), ([3], 3), ([1,3], 4), ([2,3], 5), ([1,2,3], 6)]
```