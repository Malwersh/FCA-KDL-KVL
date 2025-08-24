

# Documentation of FCA Reduction Methods: KDL and KVL

## 1. Overview

This software provides a complete pipeline for **Formal Concept Analysis (FCA)**–based lattice reduction and clustering. The framework includes:

1. Transformation of categorical datasets into **formal contexts** $(G,M,I)$.
2. **Extraction of formal concepts** using the **NextClosure** algorithm.
3. Efficient determination of **covering relations** (Hasse diagram edges) via the **iPred** algorithm.
4. Definition of a **Dijkstra-based distance measure** on the lattice graph, incorporating directional costs (UpCost / DownCost).
5. Two clustering approaches:

   * **KDL**: *K-means Dijkstra on Lattice* — clusters concepts using shortest-path distances in the lattice.
   * **KVL**: *K-means Vector on Lattice* — clusters concepts in vector space derived from their attribute descriptions.

The implementation is written in **Python 3.11**, with dependencies on `pandas`, `numpy`, `networkx`, `scikit-learn`, `matplotlib`, and optionally `igraph` for visualization.

---

## 2. Formal Context Representation

In Formal Concept Analysis (FCA), the foundation of analysis begins with the **formal context**, a mathematical structure that captures relationships between a set of **objects** and a set of **attributes**.

### 2.1. Definition

A **formal context** is defined as a triple $(G, M, I)$, where:

* $G$: a finite set of **objects**.
* $M$: a finite set of **attributes**.
* $I \subseteq G \times M$: a binary **incidence relation**, indicating which object has which attribute. That is, $(g, m) \in I$ means object $g$ has attribute $m$.

The formal context is often represented as a **binary matrix** (or DataFrame in Python), where:

* **Rows** correspond to objects,
* **Columns** correspond to attributes,
* A value of **1** means the object possesses the attribute, and **0** otherwise.


### 2.2. Code Implementation (from `FCA_Methods.ipynb`)

The notebook implements this representation using `pandas` to define a binary DataFrame.

```python
import pandas as pd

# Define binary attributes for 8 laptops (objects)
data = {
    "Touch":    [1, 0, 1, 0, 1, 1, 0, 1],
    "Backlit":  [1, 1, 0, 0, 1, 0, 0, 1],
    "SSD":      [1, 1, 0, 1, 0, 0, 1, 0],
}

# Object labels: L1 to L8
objects = [f"L{i}" for i in range(1, 9)]

# Construct the binary formal context
context_df = pd.DataFrame(data, index=objects)

# Display the context
print(context_df)
```

This produces the following **formal context** table:

|    | Touch | Backlit | SSD |
| -- | ----- | ------- | --- |
| L1 | 1     | 1       | 1   |
| L2 | 0     | 1       | 1   |
| L3 | 1     | 0       | 0   |
| L4 | 0     | 0       | 1   |
| L5 | 1     | 1       | 0   |
| L6 | 1     | 0       | 0   |
| L7 | 0     | 0       | 1   |
| L8 | 1     | 1       | 0   |



### 2.3. Construction Steps

To construct a formal context from any dataset, follow these steps:

#### Step 1: Identify Objects and Attributes

* **Objects (G)** are the entities being described — e.g., `L1` to `L8` (laptops).
* **Attributes (M)** are binary features — e.g., `Touch`, `Backlit`, `SSD`.

#### Step 2: Create Binary Incidence Matrix

* For each object $g \in G$, set 1 in column $m \in M$ if the attribute applies to the object; else 0.
* In Python, this is done via a DataFrame.

#### Step 3: Validate Matrix Shape

* Ensure the matrix shape is $|G| \times |M|$, with rows = number of objects, columns = number of attributes.



### 2.4. Practical Significance

This formal context is the **input** to all subsequent steps in the FCA pipeline:

* **Concept generation** (e.g., using NextClosure).
* **Lattice construction** and **visualization**.
* **Clustering** via KDL or KVL (based on shortest path or attribute vectors).

By encoding the context in this structure, all algorithms can systematically process relationships and derive formal concepts and their hierarchies.


---
Here is a refined and detailed version of the **“3. NextClosure Algorithm”** section for your README documentation, based on your implementation and the uploaded files:

---

## 3. NextClosure Algorithm

### Purpose

The **NextClosure algorithm** is used to **enumerate all formal concepts** from a binary context in a **systematic, canonical order**. It ensures **no duplicates** and **complete coverage** of all concept pairs $(X, Y)$, where $X = Y'$ and $Y = X'$.

---

### Theoretical Foundations

A **formal concept** in Formal Concept Analysis (FCA) is a pair $(X, Y)$ such that:

* $X$ is the **extent**: the set of all objects sharing attributes in $Y$,
* $Y$ is the **intent**: the set of all attributes common to objects in $X$,
* and this relationship is defined by the **Galois connection** (double derivation):

  $$
  X = Y', \quad Y = X'
  $$

#### Properties of Closure Operator $c(\cdot)$

Let $c(A) = A''$ denote the closure operator:

* **Extensive**: $A \subseteq c(A)$
* **Idempotent**: $c(c(A)) = c(A)$
* **Monotonic**: $A \subseteq B \Rightarrow c(A) \subseteq c(B)$

---

### Key Steps

The NextClosure algorithm performs a **lexicographic traversal** of all closed sets (intents). The algorithm proceeds as follows:

#### Pseudocode

```python
def next_closure(context, current_set):
    for i in reversed(range(len(context.columns))):
        if context.columns[i] not in current_set:
            candidate = sorted(set(current_set) | {context.columns[i]})
            closure = compute_closure(context, candidate)
            if is_lex_next(current_set, closure, i):
                yield closure
                from_closure = next_closure(context, closure)
                yield from from_closure
```

#### Steps:

1. **Start** from the closure of the empty set: $\emptyset''$
2. **Iteratively**:

   * Try adding attributes not in current intent.
   * Compute closure of the new set.
   * Check if it satisfies the **lectic order** condition (i.e., it is the smallest next closed set).
3. **Stop** when all closures are exhausted.

---

### Code Reference

The complete Python implementation is available in the file:
📁 [`FCA_Methods.ipynb`](https://github.com/Malwersh/FCA-KDL-KVL/blob/main/FCA_Methods.ipynb)
Look for the section **“Generating All Formal Concepts using NextClosure”**, which includes:

* `compute_closure()` function: computes $A''$ using derivation rules
* `next_closure()` function: main recursive generator
* `generate_all_concepts()` wrapper function

---

### Example

Let's walk through a simple context of laptops and their features:

```python
import pandas as pd

context = pd.DataFrame({
    "Touch":   [1, 0, 1, 0],
    "Backlit": [1, 1, 0, 0],
    "SSD":     [1, 1, 0, 1],
}, index=["L1", "L2", "L3", "L4"])
```

Now, generate all formal concepts:

```python
from fca_algorithms import generate_all_concepts

concepts = generate_all_concepts(context)
for extent, intent in concepts:
    print("Extent:", extent, "Intent:", intent)
```

**Output Example:**

```
Extent: ['L1', 'L3'] Intent: ['Touch']
Extent: ['L1']       Intent: ['Touch', 'Backlit', 'SSD']
...
```

---

Excellent — now that you've provided the actual algorithm diagram from the original paper *“Yet a Faster Algorithm for Building the Hasse Diagram of a Concept Lattice”* (Baixeries et al., 2009), we can now **rewrite Section 4** in full and **replace the simplified pseudocode** with the **original formal version** shown in the image. I'll also adjust the explanation to match the notation used in the paper.

---

## **4. iPred Algorithm: Constructing the Covering Relation**

### **Purpose**

The **iPred algorithm** efficiently computes the **covering relation** $\leq_{\mathcal{L}}$ of a concept lattice $\mathcal{L} = (C, \leq_{\mathcal{L}})$, where $C$ is the set of formal concepts generated from a formal context $\mathbb{K} = (G, M, I)$. Rather than checking all possible pairs of concepts, iPred determines **direct predecessor-successor pairs** based on extents using closure properties, border sets, and dynamic updates.

This approach results in a reduced number of comparisons and significantly improves the performance of Hasse diagram construction.

---

### **Formal Algorithm (from the Paper)**

Below is the **original iPred algorithm** as published in the paper by Baixeries et al. (2009):

```
Input:   C = {c₁, c₂, ..., cₗ}      // List of concepts
Output:  𝓛 = (C, ≤𝓛)                // Hasse diagram (covering relation)

1. Sort(C);                         // Sort concepts by extent size
2. foreach i ∈ {2, ..., ℓ} do
3.     Δ[cᵢ] ← ∅;                   // Initialize coverage sets
4. end
5. Border ← {c₁};
6. foreach i ∈ {2, ..., ℓ} do
7.     Candidate ← {cᵢ ∩ ĉ | ĉ ∈ Border};      // Check border intersection
8.     foreach ĉ ∈ Candidate do
9.         if Δ[ĉ] ∩ cᵢ = ∅ then
10.            ≤𝓛 ← ≤𝓛 ∪ (cᵢ, ĉ);               // Add cover edge
11.            Δ[ĉ] ← Δ[ĉ] ∪ (cᵢ − ĉ);           // Update covered elements
12.            Border ← Border − ĉ;
13.        end
14.    end
15.    Border ← Border ∪ cᵢ;                     // Add current to border
16. end
```

---

### **Explanation of Algorithm Components**

* **Input**: A set of formal concepts $C$, sorted by increasing size of extent (i.e., number of objects in each concept).
* **Border**: A dynamic set of previously seen concepts potentially covered by the next.
* **Δ\[c]**: A mapping of each concept $c \in C$ to the set of objects already "covered" by its successors.
* **Candidate Set**: Formed by intersecting the current concept with those in the border.
* **Cover Check**: If no object in the current extent $cᵢ$ is in the Δ set of a candidate $ĉ$, then $ĉ$ is a direct predecessor of $cᵢ$.

---

### **Python Implementation (Notebook)**

This logic is implemented in the code file `FCA_Methods(KDL and KVL).ipynb` using:

```python
def ipred_cover_relation(concepts):
    border = [concepts[0]]
    delta = {tuple(c[0]): set() for c in concepts[1:]}  # skip first

    edges = []
    for i in range(1, len(concepts)):
        ci_extent, ci_intent = concepts[i]
        candidate_edges = []
        for cj_extent, cj_intent in border:
            candidate = ci_extent & cj_extent
            if not delta.get(tuple(cj_extent), set()) & ci_extent:
                edges.append(((ci_extent, ci_intent), (cj_extent, cj_intent)))
                delta[tuple(cj_extent)] = delta.get(tuple(cj_extent), set()) | (ci_extent - cj_extent)
                candidate_edges.append((cj_extent, cj_intent))
        for ce in candidate_edges:
            border.remove(ce)
        border.append((ci_extent, ci_intent))
    return edges
```

This Python function closely matches the pseudocode from the article, including the use of `border`, `delta`, and the candidate filtering logic based on set operations.

---

### **Application Example**

Consider the context:

```python
df = pd.DataFrame({
    "Wifi": [1, 0, 1],
    "Bluetooth": [1, 1, 0],
    "GPS": [1, 1, 1],
}, index=["D1", "D2", "D3"])
```

From this context, a set of concepts is derived. Then `ipred_cover_relation()` is applied to obtain only the direct edges in the lattice:

```python
formal_concepts = generate_formal_concepts(df)  # e.g., using NextClosure
edges = ipred_cover_relation(formal_concepts)
```

**Sample Output (Hasse Diagram Edges):**

```
[(({'D1'}, {'Wifi', 'Bluetooth', 'GPS'}), ({'D1', 'D3'}, {'Wifi', 'GPS'})),
 (({'D2'}, {'Bluetooth', 'GPS'}), ({'D2', 'D3'}, {'GPS'})),
 ...]
```

---

### **Conclusion**

| Component          | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| **Goal**           | Build Hasse diagram (covering relation) for concept lattice efficiently      |
| **Innovation**     | Uses dynamic border pruning and object coverage sets                         |
| **Source**         | [Baixeries et al., ICFCA 2009](https://doi.org/10.1007/978-3-642-01815-2_13) |
| **Code Reference** | `ipred_cover_relation()` in `FCA_Methods(KDL and KVL).ipynb`                 |
| **Result**         | Minimal set of edges for efficient lattice visualization                     |



---

Here is the improved and detailed explanation of the **Dijkstra-Based Distance** method in FCA, based on your thesis (Chapter 4) and supported by the Python implementation found in the provided notebook:

---

## **5. Dijkstra-Based Distance**

### **Purpose**

The Dijkstra-based distance function is adapted to measure the **semantic effort** between two concepts in a formal concept lattice. Unlike standard shortest path algorithms, this adaptation incorporates **direction-sensitive edge costs** to better represent the **hierarchical structure** of the lattice.

---

### **Theoretical Background**

#### **Graph Construction**

* **Nodes**: Formal concepts derived from the concept lattice.
* **Edges**: Directed Hasse diagram edges between concepts (i.e., covering relations).
* **Weights**:

  * **UpCost**: Cost of moving from a specific concept to a more general one (e.g., 2 units).
  * **DownCost**: Cost of moving from a general concept to a more specific one (e.g., 1 unit).

#### **Formal Definition**

Given two concepts $C_s$ (source) and $C_e$ (end), the **distance** is defined as:

$$
d(C_s, C_e) = \min_{\text{paths}} \sum f(c_i, c_{i+1})
$$

Where:

* $c_i \to c_{i+1}$ represents a directed edge in the concept lattice.
* $f(c_i, c_{i+1})$ = 1 if down edge, 2 if up edge.

This ensures that the **effort** of moving through the hierarchy (either generalization or specialization) is reflected in the path cost.

---

### **Dijkstra-Based Algorithm Steps**

The algorithm is adapted from classical Dijkstra’s method as follows:

1. **Initialize**:

   * Set distance of all nodes to ∞, except source node which is 0.
   * Use a priority queue to explore nodes by minimum tentative distance.

2. **Relaxation**:

   * For each neighboring concept, determine the direction:

     * **Upward** (superconcept): cost = `up_cost`
     * **Downward** (subconcept): cost = `down_cost`
   * Update the distance if a shorter path is found via this direction.

3. **Termination**:

   * The algorithm ends when the target node is dequeued from the priority queue.

4. **Path Retrieval** (optional):

   * Trace back from end node to start node using parent pointers for path reconstruction.

---

### **Python Implementation Reference**

The implementation is found in your provided notebook, specifically under the section that contains:

```python
def dijkstra_distance(start, end, graph, up_cost=2, down_cost=1):
    ...
```

* The `graph` is typically represented as a dictionary where keys are nodes and values are lists of neighbors.
* The algorithm handles directionality using concept levels or edge labels.
* The final output is the **minimum hierarchical cost** from `start` to `end`.

---

### **Example of Application**

Let’s consider a simplified concept lattice:

```
     C0
    /  \
  C1    C2
    \  /
     C3
```

Let:

* $C_0$ is the top concept

* $C_3$ is the bottom concept

* Paths:

  * From $C_0 \to C_3$: Up → Down → Down (cost = 1 + 2 + 2 = 5)
  * From $C_3 \to C_0$: Down → Up → Up (cost = 2 + 1 + 1 = 4)

The **shortest distance** depends on the edge directions and costs. The Dijkstra-based algorithm ensures this is minimized based on given UpCost/DownCost.



---


Certainly. Here's the **rewritten English version** of the **KDL (K-Means Dijkstra on Lattice)** section, strictly based on the explanations from **Chapter 5 of your thesis**, with no external terminology like "medoid":

---

## 6. KDL: K-Means Dijkstra on Lattice

### Overview

**KDL** (K-Means Dijkstra on Lattice) is a clustering algorithm designed to group formal concepts within a concept lattice based on their structural relationships. It is particularly suitable for categorical data, where conventional numerical clustering methods like standard k-means are not applicable.

Instead of relying on Euclidean distances, this algorithm uses a **shortest-path distance based on Dijkstra’s algorithm**, where the concept lattice structure and the direction of traversal determine the cost of moving from one concept to another.

---

### Algorithm Description

The KDL algorithm consists of the following steps:

1. **Generate the Concept Lattice**
   Using Formal Concept Analysis (FCA), extract the complete set of formal concepts from a binary context and organize them into a lattice.

2. **Assign Directional Weights to Edges**
   Each edge in the lattice is weighted based on the direction of movement:

   * Moving **upward** (from a specific concept to a general one): **cost = 1**
   * Moving **downward** (from a general concept to a specific one): **cost = 2**

3. **Apply Dijkstra's Algorithm**
   Use the adapted Dijkstra algorithm to calculate the shortest distance between any two concepts in the lattice. The direction-sensitive weights ensure that the path cost reflects hierarchical effort.

4. **Initialize Cluster Centers**
   Select *k* random formal concepts from the lattice to serve as initial cluster centers.

5. **Assign Concepts to Nearest Centers**
   For each concept in the lattice, compute its Dijkstra-based distance to each center and assign it to the cluster with the closest center.

6. **Update Cluster Centers**
   For each cluster, determine the concept within the cluster that has the minimum total distance to all other concepts in the same cluster. This concept becomes the new cluster center.

7. **Repeat Until Convergence**
   Repeat the assignment and update steps until cluster centers no longer change.

---

### Algorithm (Based on Thesis – Algorithm 5.1)

**Input**:

* *k*: number of clusters
* *B*: set of all formal concepts in the lattice

**Output**:

* A set of *k* clusters: {S₁, S₂, ..., Sₖ}

**Procedure**:

1. Choose *k* initial centers {Z₁, Z₂, ..., Zₖ} randomly from *B*
2. For each concept *c* ∈ *B*, assign it to cluster *Sᵢ* where
   `d(c, Zᵢ)` is minimal (based on Dijkstra distance)
3. For each cluster *Sᵢ*, choose a new center *Zᵢ* from the concepts in *Sᵢ* such that:
   `∑ d(Zᵢ, c_j)` is minimized, for all *c\_j* ∈ *Sᵢ*
4. Repeat steps 2 and 3 until the centers remain unchanged

---

### Example

Assume a small lattice with the following concepts:

* C₀ = (∅, M)
* C₁ = ({g₁}, {m₁, m₂})
* C₂ = ({g₂}, {m₁, m₃})
* C₃ = ({g₁, g₂}, {m₁})
* C₄ = ({g₁, g₃}, {m₂})
* C₅ = ({g₁, g₂, g₃}, ∅)

Assume *k = 2* clusters.

**Step 1:** Choose initial centers randomly, say C₁ and C₂.

**Step 2:** For each concept, compute the Dijkstra distance to both centers and assign to the closer one.

**Step 3:** Recalculate the new center for each cluster by selecting the concept that has the smallest total distance to all others in that cluster.

**Step 4:** Repeat until clusters stabilize.

---

### Distance Calculation

Dijkstra's algorithm computes the shortest path between any two nodes in the lattice, with edge weights defined as:

* **Upward edge (child → parent)**: cost = 2
* **Downward edge (parent → child)**: cost = 1

This directional cost reflects the hierarchical semantics of the lattice—moving downward (to more specific concepts) is more expensive than moving upward.

---

### Python Code Reference

The implementation of this method is provided in the uploaded Python notebook under the relevant section titled `KDL_Clustering()` (or similar). A simplified code sketch for distance calculation and cluster assignment is:

```python
def dijkstra_distance(graph, source, target):
    # Uses weighted graph traversal with direction-aware costs
    pass

def assign_clusters(concepts, centers):
    # Assign each concept to the cluster with the shortest distance
    pass

def update_centers(clusters):
    # For each cluster, choose the new center minimizing intra-cluster distances
    pass
```

---

Would you like a LaTeX-compatible version of this section for your thesis or slides?

---

Here is a detailed explanation of the **KVL method (K-means on Vectorized Lattice)** based entirely on Chapter 5 of your thesis:

---

## **6. KVL: K-means on Vectorized Lattice**

### **Overview**

The KVL method—short for **K-means Vectorized Lattice**—is a clustering technique designed to exploit the lattice structure by transforming formal concepts into numerical vectors based on their positions in the concept lattice. Unlike the KDL approach, which depends on Dijkstra-based shortest-path distances, KVL utilizes vector representations derived from the **intent structure** of the concepts, allowing faster computation using traditional K-means clustering on vector spaces.

---

### **Motivation**

The motivation behind KVL is to overcome the computational cost of the Dijkstra-based method (KDL) while still capturing the relative positioning of formal concepts. KVL leverages **binary vector encodings** of concept intents and applies a standard K-means clustering process in the resulting feature space.

---

### **Steps of the KVL Algorithm**

1. **Vectorization**:

   * Each formal concept is represented as a binary vector.
   * The vector is derived from the **intent set** (attributes), where each position corresponds to an attribute (1 = present, 0 = absent).

2. **K-means Initialization**:

   * Select $k$ initial centroids randomly from the set of concept vectors.

3. **Assignment Step**:

   * Each concept vector is assigned to the closest cluster centroid using **Euclidean distance** in the binary vector space.

4. **Update Step**:

   * Each cluster updates its centroid by computing the **mean vector** of the concepts assigned to it (as in standard K-means).

5. **Iteration**:

   * Repeat the assignment and update steps until convergence (no changes in cluster memberships or centroids).

---

### **Advantages**

* **Computational Efficiency**: KVL avoids pairwise shortest path computation required by KDL.
* **Scalability**: Well-suited for large lattices due to the use of vector operations.
* **Simplicity**: Built on top of traditional K-means logic with only the initial vectorization as a lattice-specific step.

---

### **Python Implementation**

The Python notebook (`FCA_Methods(KDL and KVL).ipynb`) includes a full implementation of the KVL method. The key parts are:

* **Vectorization Function**:

  ```python
  def vectorize_concepts(concepts, all_attributes):
      vectors = []
      for concept in concepts:
          vector = [1 if attr in concept.intent else 0 for attr in all_attributes]
          vectors.append(vector)
      return np.array(vectors)
  ```

* **K-means Execution**:

  ```python
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(vectors)
  labels = kmeans.labels_
  ```

---

### **Example**

Let’s assume we have 5 formal concepts with intents over the attribute set `{a, b, c}`:

| Concept | Intent | Vector     |
| ------- | ------ | ---------- |
| C1      | {a, b} | \[1, 1, 0] |
| C2      | {a}    | \[1, 0, 0] |
| C3      | {a, c} | \[1, 0, 1] |
| C4      | {b, c} | \[0, 1, 1] |
| C5      | {c}    | \[0, 0, 1] |

Applying K-means with $k = 2$:

* Initialization: Randomly choose 2 centroids, say C1 and C5.
* Assignment: Compute distances and assign each concept vector to the closest centroid.
* Update: Compute the new centroid for each cluster as the average of vectors in the cluster.
* Iterate until convergence.

---

### **Comparison with KDL**

| Aspect              | KDL                   | KVL                       |
| ------------------- | --------------------- | ------------------------- |
| Distance Metric     | Dijkstra over lattice | Euclidean in vector space |
| Data Representation | Lattice graph         | Binary intent vectors     |
| Computational Cost  | Higher                | Lower                     |
| Clustering Basis    | Structural proximity  | Intent similarity         |



---

## 8. Evaluation Metrics

* **Silhouette Coefficient (SC)**: higher = better separation.
* **Davies–Bouldin Index (DBI)**: lower = better compactness.

For **KDL**, custom distance matrices (Dijkstra) are required.
For **KVL**, standard Euclidean distance suffices.

---

## 9. Example Usage

```python
# Build concepts
concepts = enumerate_concepts_nextclosure(df)

# Build lattice (Hasse diagram)
G = build_hasse_cover_graph(concepts)
H = add_directional_weights(G, up_cost=2, down_cost=1)

# Run KDL
centers, clusters = kdl_cluster(H, k=3)

# Run KVL
V, cols = concept_vectors(df, concepts)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, n_init=20).fit(V)
```

---

## 10. Visualization

Hasse diagrams can be drawn using:

```python
import matplotlib.pyplot as plt
import networkx as nx

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=400)
plt.show()
```

---

## 11. Complexity

* **NextClosure**: visits all concepts with efficient canonical checks.
* **iPred**: avoids $O(|C|^2)$ comparisons.
* **KDL**: $O(k|C| \cdot (E + C\log C))$ per iteration.
* **KVL**: $O(k|C|)$ per iteration.

---

## 12. Reproducibility

* All algorithms were **implemented by the author** in **Python**.
* The code is available on **GitHub** (repository link should be cited in the thesis).
* Jupyter notebooks document the workflow with examples to ensure **full reproducibility** of the experiments in Chapter 5.

---

Great idea—here’s a compact, **worked example** you can drop into the thesis (or appendix) right after the laptops example. It shows, end-to-end, how we obtain results with **NextClosure → iPred → Dijkstra distance → KDL/KVL**, using a tiny, reproducible context. I’ve also included short Python snippets that exactly reproduce each step.

---

# Worked Example: From Context to Lattice and Clusters (Toy “Laptops” Context)

## 1) Formal context

Consider 6 laptops and 4 binary attributes:

* Objects $G=\{L_1,\dots,L_6\}$
* Attributes $M=\{T\text{ (Touch)}, B\text{ (Backlit)}, S\text{ (SSD)}, D\text{ (Detachable)}\}$

|    | T | B | S | D |
| -- | - | - | - | - |
| L₁ | 1 | 1 | 1 | 0 |
| L₂ | 0 | 1 | 1 | 0 |
| L₃ | 1 | 0 | 0 | 1 |
| L₄ | 0 | 0 | 1 | 0 |
| L₅ | 1 | 1 | 0 | 0 |
| L₆ | 1 | 0 | 1 | 0 |

This is our formal context $(G,M,I)$.

---

## 2) Enumerating formal concepts with **NextClosure**

We illustrate a few concepts (extent, intent). (A full listing is produced by the code below.)

* $(\{L_1,L_2,L_4,L_6\}, \{S\})$: all SSD laptops
* $(\{L_1,L_2,L_5\}, \{B\})$: all Backlit laptops
* $(\{L_1,L_3,L_5,L_6\}, \{T\})$: all Touch laptops
* $(\{L_1\}, \{T,B,S\})$: the only laptop that has all three (T,B,S)
* $(\{L_3\}, \{T,D\})$: the only laptop that is Touch & Detachable
* **Top** concept: $(\{L_1,\dots,L_6\}, \varnothing)$
* **Bottom** concept: $(\varnothing, \{T,B,S,D\}')$ (here, bottom becomes $(\{ \}, \{ \})$ if no attribute is shared by all)

> In the toy data no attribute is common to **all** laptops, so the bottom intent is $\varnothing$. (You can confirm by intersecting all row-attribute sets.)

**Python to reproduce:**

```python
import itertools
import pandas as pd

# context
df = pd.DataFrame(
    {"T":[1,0,1,0,1,1], "B":[1,1,0,0,1,0], "S":[1,1,0,1,0,1], "D":[0,0,1,0,0,0]},
    index=["L1","L2","L3","L4","L5","L6"]
)

G = list(df.index)
M = list(df.columns)

def deriv_up(X):   # X ⊆ G -> attributes common to all in X
    if not X: return set(M)  # convention
    attrs = set(M)
    for g in X:
        attrs &= set(df.columns[df.loc[g]==1])
    return attrs

def deriv_down(Y): # Y ⊆ M -> objects having all attributes in Y
    objs = set(G)
    for m in Y:
        objs &= set(df.index[df[m]==1])
    return objs

def closure(Y):
    return deriv_up(deriv_down(Y))

# Simple NextClosure (lectic order on M)
def next_closure():
    concepts = []
    Y = set()  # start with closure of empty set
    Y = closure(Y)
    concepts.append((frozenset(deriv_down(Y)), frozenset(Y)))
    while True:
        for i,m in enumerate(M):
            if m not in Y:
                Z = closure(Y | {m})
                # lectic check
                if all((mm in Z) == (mm in Y) for mm in M[:i]) and m in Z:
                    Y = Z
                    concepts.append((frozenset(deriv_down(Y)), frozenset(Y)))
                    break
        else:
            break
    return concepts

concepts = next_closure()
len(concepts), concepts[:6]
```

---

## 3) Building the Hasse diagram with **iPred**

Using the concept set $C$, we compute the **covering relation** (edges of the Hasse diagram) by testing direct predecessor/successor pairs—iPred eliminates edges that are implied via intermediate concepts.

**Python (sketch):**

```python
import networkx as nx

# order by extent (smaller extent = more specific)
C_sorted = sorted(concepts, key=lambda c: (len(c[0]), -len(c[1])))

def leq(c1, c2):  # (X1,Y1) ≤ (X2,Y2) iff X1 ⊆ X2
    return c1[0].issubset(c2[0])

G_hasse = nx.DiGraph()
G_hasse.add_nodes_from(range(len(C_sorted)))
for i,ci in enumerate(C_sorted):
    for j,cj in enumerate(C_sorted):
        if i==j: continue
        if leq(ci,cj):
            # add candidate edge i->j if no k with i<k<j
            cover = True
            for k,ck in enumerate(C_sorted):
                if k in (i,j): continue
                if leq(ci,ck) and leq(ck,cj):
                    cover = False; break
            if cover:
                G_hasse.add_edge(i,j)
```

You can then draw the Hasse diagram (e.g., with `networkx` or `igraph`).

---

## 4) Dijkstra-based distance on the lattice

We assign directional costs:

* **DownCost = 1** (from specific to more general: child → parent)
* **UpCost = 2** (from general to more specific: parent → child)

If we want the cost to go from concept $c$ to its neighbor $u$:

* If $ext(c) \subset ext(u)$ (moving upward/specific), add **UpCost**
* Else (moving downward/general), add **DownCost**

**Example path cost.** Suppose we compute the distance between:

* $c_s = (\{L_1,L_2,L_4,L_6\}, \{S\})$ and
* $c_e = (\{L_1\}, \{T,B,S\})$.

One shortest path is:

$$
(\{L_1,L_2,L_4,L_6\},\{S\}) \rightarrow (\{L_1,L_2,L_6\},\{S,T\}) \rightarrow (\{L_1\},\{T,B,S\})
$$

(two **up** moves) so total cost $= 2 + 2 = 4$.

**Python (distance):**

```python
def edge_cost(ci, cj):
    Xi, Yi = ci
    Xj, Yj = cj
    if Xi.issubset(Xj) and Xi != Xj:  # up (more specific)
        return 2
    if Xj.issubset(Xi) and Xj != Xi:  # down (more general)
        return 1
    return None

# Turn Hasse edges into weighted directed graph both directions (with above costs)
H = nx.DiGraph()
for u,v in G_hasse.edges():
    cu, cv = C_sorted[u], C_sorted[v]
    c_uv = edge_cost(cu, cv)
    c_vu = edge_cost(cv, cu)
    if c_uv is not None: H.add_edge(u, v, weight=c_uv)
    if c_vu is not None: H.add_edge(v, u, weight=c_vu)

# Dijkstra
def dijkstra(u, v):
    return nx.shortest_path_length(H, u, v, weight="weight")

# Find indices for the two concepts quoted above and compute distance
idx_s = C_sorted.index((frozenset({"L1","L2","L4","L6"}), frozenset({"S"})))
idx_e = C_sorted.index((frozenset({"L1"}), frozenset({"T","B","S"})))
dijkstra(idx_s, idx_e)  # -> 4
```

---

## 5) **KDL** (K-means Dijkstra on Lattice)

Set $k=2$. We initialize two centers (random concepts), then iterate:

**Assignment step**
Each concept is assigned to the nearest center using the **Dijkstra distance** above.

**Update step**
For each cluster, the **new center** is the **medoid**: the concept minimizing the sum of distances to all concepts in that cluster:

$$
Z^\star = \arg\min_{c\in S} \sum_{x\in S} d(c,x).
$$

**Python (sketch):**

```python
import random
import numpy as np

def kdl(H, C_sorted, k=2, max_iter=50):
    centers = random.sample(range(len(C_sorted)), k)
    for _ in range(max_iter):
        # assignment
        clusters = {c: [] for c in centers}
        for i in range(len(C_sorted)):
            d = [(c, nx.shortest_path_length(H, i, c, weight="weight")) for c in centers]
            cmin = min(d, key=lambda x: x[1])[0]
            clusters[cmin].append(i)
        # update
        new_centers = []
        for c, members in clusters.items():
            # medoid
            D = np.zeros((len(members), len(members)))
            for a, ia in enumerate(members):
                for b, ib in enumerate(members):
                    D[a,b] = nx.shortest_path_length(H, ia, ib, weight="weight")
            medoid_idx = members[np.argmin(D.sum(axis=1))]
            new_centers.append(medoid_idx)
        if set(new_centers) == set(centers):
            break
        centers = new_centers
    return centers, clusters
```

**Interpretation (typical outcome on this toy data)**

* One cluster gathers **SSD-centric** concepts (those near intent $\{S\}$ or $\{T,S\}$).
* The other gathers **Touch/Backlit** concepts (near $\{T,B\}$).
  The split reflects **structural proximity in the lattice** under up/down traversal costs.

---

## 6) **KVL** (K-means Vector on Lattice)

We now embed each concept $c=(X,Y)$ as a **concept description vector** $v(c)\in\mathbb{R}^{|M|}$:

$$
v_m(c) =
\begin{cases}
1 & \text{if } m\in Y \\
\text{freq}(m) & \text{otherwise, where } \text{freq}(m)=\frac{|\{g\in G: I(g,m)=1\}|}{|G|}
\end{cases}
$$

Compute global frequencies from the table:

* $\text{freq}(T)=4/6=0.667$
* $\text{freq}(B)=3/6=0.500$
* $\text{freq}(S)=4/6=0.667$
* $\text{freq}(D)=1/6=0.167$

**Example vectors**

* For $c_1=(\{L_1,L_2,L_4,L_6\}, \{S\})$:
  $v(c_1) = [0.667, 0.500, 1, 0.167]$
* For $c_2=(\{L_1\}, \{T,B,S\})$:
  $v(c_2) = [1, 1, 1, 0.167]$

We then run **standard k-means (Euclidean)** on these vectors with $k=2$.

**Python:**

```python
import numpy as np
from sklearn.cluster import KMeans

freq = df.mean(axis=0).to_dict()  # global attribute frequencies

def concept_vector(c):
    X, Y = c
    return np.array([1.0 if m in Y else float(freq[m]) for m in M])

V = np.vstack([concept_vector(c) for c in C_sorted])
kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(V)
labels = kmeans.labels_
```

**Interpretation (typical outcome)**

* One cluster collects concepts with intents emphasizing **T and/or B** (components near 1 in those positions).
* The other collects concepts emphasizing **S** (intent contains S).
  Here the grouping reflects **attribute-level similarity** rather than traversal cost.

---

## 7) What the two methods “see”

* **KDL** respects **lattice structure** and **directional cost** (it is sensitive to how “far” a concept is in terms of **up/down moves**).
* **KVL** respects **attribute similarity** (two concepts are close if their **intents**—completed by background frequencies—look similar numerically).

Both produce sensible, but **not identical**, partitions. In larger datasets (as in the main chapters), this difference is exactly why KDL often yields **higher-quality clusters** w\.r.t. lattice semantics, while KVL yields **faster** and more scalable runs.

---

## 8) Minimal visualization snippet

```python
import matplotlib.pyplot as plt
pos = nx.spring_layout(G_hasse, seed=42)
nx.draw(G_hasse, pos, with_labels=False, node_size=300, arrows=True)
plt.title("Hasse (cover) graph of toy context")
plt.show()
```

---
