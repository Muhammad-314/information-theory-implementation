# information-theory-implementation

An implementation-first exploration of **information theory through code**.

This repository is an educational project I built to understand information theory more deeply by implementing concepts myself, experimenting with probability distributions, checking mathematical relationships numerically, and observing how the quantities behave under different conditions.

This is **not intended as research, a novel contribution, a production-ready library, or a comprehensive implementation of information theory**. The goal was simply to learn the subject more seriously by moving back and forth between the mathematics, the code, and experiments.

## What this project explores

> **Math rendering:** Equations use standard LaTeX delimiters (`\(...\)` for inline math and `$$...$$` for display math).

The notebook progressively builds up several fundamental information-theoretic concepts:

1. **Self-information**
2. **Shannon entropy**
3. **Entropy and class imbalance**
4. **Joint entropy**
5. **Conditional entropy**
6. **Mutual information**
7. **Cross-entropy**
8. **KL divergence**
9. **KL divergence between Gaussian distributions**
10. **Jensen–Shannon divergence**
11. **An integrated experiment connecting these ideas**

The implementations are intentionally small and explicit. Rather than hiding the mathematics behind existing libraries, the project tries to make the formulas visible in the code.

---

## Concepts implemented

### 1. Self-information

For an event with probability \(p(x)\), self-information is implemented as

$$\nI(x) = -\log_2 p(x)\n$$

This provides the basic intuition behind information theory: **rare events carry more information than common events**.

The notebook experiments with different probabilities to observe this relationship directly.

---

### 2. Shannon entropy

For a discrete random variable \(X\),

$$\nH\(X\) = -\sum_x p(x)\log_2 p(x)\n$$

Entropy measures the average uncertainty of a probability distribution.

The notebook explores:

* deterministic distributions
* uniform distributions
* increasingly skewed distributions
* entropy as the number of possible outcomes changes
* the relationship between probability concentration and uncertainty

All entropy values are expressed in **bits**.

---

### 3. Entropy and class imbalance

The project also looks at entropy through the lens of binary/class distributions.

This helps connect the mathematical definition of entropy with a more intuitive setting:

* balanced classes have higher uncertainty
* highly imbalanced classes have lower uncertainty
* a completely deterministic class distribution has zero entropy

This section is useful for building intuition before moving into joint distributions.

---

### 4. Joint entropy

For two discrete random variables \(X\) and \(Y\),

$$\nH(X,Y) = -\sum_{x,y} p(x,y)\log_2 p(x,y)\n$$

The implementation works with joint probability tables and uses them to explore how uncertainty behaves when considering two variables together.

---

### 5. Conditional entropy

Conditional entropy is implemented using

$$\nH(Y|X) = H(X,Y) - H\(X\)\n$$

The notebook uses joint distributions to investigate how observing one variable can reduce uncertainty about another.

This provides the bridge between entropy and mutual information.

---

### 6. Mutual information

Mutual information is implemented as

$$\nI(X;Y) = H\(X\) + H\(Y\) - H(X,Y)\n$$

It is used to measure how much information one random variable contains about another.

The notebook includes examples involving:

* perfectly dependent variables
* independent variables
* partially dependent variables
* empirical joint distributions estimated from samples

An empirical joint-distribution helper is also implemented so that mutual information can be computed from generated discrete data rather than only from manually specified probability tables.

---

### 7. Cross-entropy

Cross-entropy is implemented as

$$\nH(P,Q) = -\sum_x P(x)\log_2 Q(x)\n$$

The project explores the connection between cross-entropy, entropy, and probability predictions.

It also includes a simple `softmax` implementation and uses predicted class probabilities to connect the information-theoretic definition of cross-entropy with a familiar machine-learning classification setting.

One of the experiments varies the probability assigned to the correct class and visualizes how the cross-entropy changes.

---

### 8. KL divergence

Kullback–Leibler divergence is implemented as

$$\nD_{KL}(P\|Q) = \sum_x P(x)\log_2\frac{P(x)}{Q(x)}\n$$

The notebook explores several important properties:

* \(D_{KL}(P|P)=0\)
* KL divergence is non-negative
* KL divergence is not symmetric
* increasingly incorrect distributions produce larger divergence
* the relationship

$$\nD_{KL}(P\|Q) = H(P,Q)-H(P)\n$$

is verified numerically.

There are also randomized checks of basic KL properties.

---

### 9. KL divergence between Gaussian distributions

The project extends KL divergence from discrete distributions to continuous probability densities.

A one-dimensional Gaussian PDF is implemented explicitly, followed by a numerical approximation of

$$\nD_{KL}(P\|Q) = \int p(x)\log_2\frac{p(x)}{q(x)}\,dx\n$$

The experiments investigate the effect of:

* mean shifts
* variance changes
* simultaneous mean and variance changes

The numerical integration is then compared against the closed-form KL divergence between two one-dimensional Gaussian distributions.

This was particularly useful for seeing how an abstract divergence measure behaves geometrically as two distributions move apart.

---

### 10. Jensen–Shannon divergence

Jensen–Shannon divergence is implemented using the midpoint distribution

[
M = \frac{P+Q}{2}
]

and

$$\nJS(P,Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)\n$$

The notebook also implements the equivalent entropy formulation:

$$\nJS(P,Q) = H(M) - \frac{1}{2}H(P) - \frac{1}{2}H(Q)\n$$

The two implementations are compared numerically.

The experiments also illustrate an important contrast with KL divergence:

* KL is generally asymmetric
* JS is symmetric
* JS remains bounded for discrete distributions
* JS behaves more gently under severe distribution mismatch

---

## Final integrated experiment

The final section brings several of the earlier ideas together in a small simulated classification/information-theory experiment.

A binary variable \(Y\) is generated, after which two different observations are constructed:

* \(X_1\): a noisy version of \(Y\)
* \(X_2\): an independent random variable

The notebook then estimates quantities such as:

$$
I(X_1;Y)
$$

and

$$
I(X_2;Y)
$$

to observe the difference between an informative signal and an unrelated one.

It also computes conditional entropy:

$$
H(Y|X_1)
$$

and

$$
H(Y|X_2)
$$

and compares different probabilistic predictions using:

* cross-entropy
* KL divergence
* Jensen–Shannon divergence

This serves as a small end-to-end experiment connecting the mathematical definitions developed throughout the notebook.

---

## Implementation philosophy

The main idea behind the project was:

> **Don't just read the formula—implement it, generate examples, and see what it actually does.**

For that reason, the notebook deliberately uses fairly direct implementations with NumPy rather than relying on specialized information-theory libraries.

The workflow throughout the notebook is roughly:

```text
mathematical definition
        ↓
small Python implementation
        ↓
simple examples
        ↓
edge cases / properties
        ↓
numerical experiments
        ↓
plots and observations
```

This makes the code somewhat more verbose than a production implementation would need to be, but that is intentional. The purpose was learning and experimentation.

---

## Numerical considerations

The implementations use small numerical safeguards such as probability clipping around logarithms:

```python
eps = 1e-12
```

This prevents expressions involving `log(0)` from producing numerical problems.

That also means the implementations should **not** be interpreted as perfectly general-purpose numerical routines. For example, mathematically,

[
D_{KL}(P|Q)
]

can be infinite when \(P(x)>0\) while \(Q(x)=0\), whereas clipping the probability produces a large finite numerical value instead.

Similarly, the numerical Gaussian KL calculation integrates over a finite interval and therefore approximates the continuous integral.

These choices are appropriate for the educational experiments here, but would need more careful treatment in a production-quality numerical library.

---

## Tools used

The project primarily uses:

* **Python**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

The implementations are contained in the notebook:

```text
Information_Theory_Implementation.ipynb
```

---

## Running the notebook

Clone the repository and open the notebook with Jupyter:

```bash
git clone https://github.com/<your-username>/information-theory-implementation.git
cd information-theory-implementation

jupyter notebook
```

Then open:

```text
Information_Theory_Implementation.ipynb
```

The notebook is intended to be read and executed sequentially because later experiments build on functions introduced earlier.

---

## What this project is — and isn't

### It is

* a personal educational project
* an implementation-based way of studying information theory
* a collection of small experiments
* a way to connect equations with numerical behavior
* a record of learning through coding

### It isn't

* a research project
* a claim of novelty
* a new information-theoretic method
* a production-ready information theory library
* a comprehensive implementation of the field
* a benchmark or authoritative reference implementation

The code is primarily valuable as a **learning artifact**.

---

## Possible future directions

If I continue expanding this project, some natural directions would be:

* discrete entropy estimation from larger datasets
* channel capacity
* binary symmetric channels
* data processing inequality experiments
* Fano's inequality
* coding theory
* Huffman coding
* arithmetic coding
* rate-distortion theory
* entropy estimation methods
* differential entropy
* multivariate Gaussian KL divergence
* information-theoretic connections to machine learning
* more systematic numerical/property-based tests

These are possibilities rather than claims about the current scope of the repository.

---

## Final note

This repository exists mainly because **implementing the mathematics made the concepts easier to understand**.

The goal wasn't to build something novel. It was to take ideas that can otherwise remain abstract—entropy, uncertainty, information, dependence, divergence, and probabilistic prediction—and make them concrete through code and experiments.

If the notebook makes even a few of those ideas more intuitive, then it served its purpose.
