# Prior-Analysis
### **Project Proposal: Redefining Atmospheric Weighting Functions using Principal Component Analysis for Optimal Filter Selection**

**1. Project Overview**

The primary objective of this project is to develop a physically representative weighting function for ground-based infrared atmospheric sounding. This will be achieved by moving away from the single-layer perturbation model and adopting a methodology based on the Principal Component Analysis (PCA) of a large dataset of atmospheric profiles. By understanding the radiance changes associated with the most significant modes of atmospheric variability, we can establish a robust, data-driven foundation for selecting the optimal number and spectral placement of infrared filters for the retrieval instrument.

**2. Technology Stack**

To implement the functional modules of this project, we will rely on the Python ecosystem, which is exceptionally well-suited for scientific computing and data analysis.

*   **Core Libraries:**
    *   **NumPy:** The fundamental package for numerical computation. It will be used for all array and matrix operations, including handling profiles, spectra, and PCA results.
    *   **Scikit-learn:** A premier machine learning library. Its `PCA` implementation is efficient and user-friendly, making it ideal for the dimensionality reduction task.
    *   **Matplotlib & Seaborn:** The standard libraries for data visualization. We will use them to plot the principal components, the explained variance, and the statistical distributions of the coefficients.
    *   **SciPy:** A library for scientific and technical computing. Its `stats` module can be used for fitting probability density functions to the coefficient distributions and for more advanced statistical analysis.

*   **Functional Module Breakdown:**
    *   **Module 1 (PCA):** Realized using `scikit-learn.decomposition.PCA`.
    *   **Module 2 (Statistical Analysis):** Realized using `NumPy` for histograms and `Matplotlib/Seaborn` for visualization.
    *   **Module 3 (Bin Size Selection):** A heuristic process guided by the visualizations from Module 2.
    *   **Module 4 (Representative Profile Selection):** Realized using `NumPy` for distance calculations (e.g., Euclidean distance).
    *   **Module 5 (Profile Perturbation):** Realized using `NumPy` for the linear algebra required to reconstruct profiles.

The modules are sequential. The output of the PCA module (coefficients and components) is the direct input for the statistical analysis and profile selection modules. The results from all preceding modules are then used in the final perturbation module.

**3. Detailed Implementation Plan**

Here is a step-by-step plan to execute the project. Each step is designed to be a self-contained task.

---

### **Step 1: Data Loading and Preparation**

**Objective:** Load the 888 atmospheric profiles and format them into matrices suitable for PCA.

**Description:** We will begin by loading the temperature and dew point profiles. Assuming each of the 888 profiles has 21 vertical layers, we will create two separate matrices: one for temperature and one for dew point.

*   `X_T`: A matrix of size (888, 21), where each row is a temperature profile.
*   `X_Td`: A matrix of size (888, 21), where each row is a dew point profile.

It is crucial to "center" the data by subtracting the mean profile from each individual profile. This is a standard prerequisite for PCA.

---

### **Step 2: Perform Principal Component Analysis**

**Objective:** Decompose the profile variations into a set of orthogonal principal components (PCs) and determine how many PCs are needed.

**Description:** We will apply PCA to the centered temperature and dew point matrices separately. PCA will yield a set of eigenvectors (the principal components, or in atmospheric science, Empirical Orthogonal Functions - EOFs) and their corresponding eigenvalues (which represent the variance captured by each component).

A key task here is to decide how many components (`k`) to retain. We will do this by examining the "explained variance ratio." We will plot the cumulative explained variance and select the number of components that captures a high percentage (e.g., 95% or 99%) of the total variance in the original profiles. This ensures our reduced model is a faithful representation of the original data.

The transformation is as follows:
$` C = X_{centered} \cdot W `$, Where:
*   $`C`$ is the matrix of principal component coefficients (or scores).
*   $`X_{centered}`$ is the (888, 21) centered data matrix.
*   $`W`$ is the matrix of principal components (eigenvectors), with shape (21, $`k`$).

---

### **Step 3: Statistical Analysis of PCA Coefficients**

**Objective:** Understand the statistical distribution of the coefficients for each retained principal component.

**Description:** For each of the `k` principal components, we have 888 coefficients. We will create histograms for the coefficients of each component to visualize their probability density function (PDF). This tells us the range and likelihood of different values for each coefficient. This step is crucial for defining a meaningful "differential element" of a coefficient.

We also need to choose a bin size for the histograms. A good starting point is to use an established rule, like Freedman-Diaconis or Sturges' formula, which balances capturing detail against noise. We can then adjust it visually to ensure we don't miss any important features in the distribution.

---

### **Step 4: Select a Representative Atmospheric Profile**

**Objective:** Choose a single, real-world temperature-dewpoint profile pair from the 888 samples to serve as a baseline for perturbation analysis.

**Description:** A common and effective method is to select the profile that is "closest" to the mean of the dataset. While the mean profile itself might be non-physical, there will be an observed profile that is most similar to it. We can find this by calculating the Euclidean distance in the coefficient space. The profile whose coefficients are closest to the origin (the mean, which is zero in centered coefficient space) is our most representative profile. This profile corresponds to a real measurement and is statistically central.

---

### **Step 5: Generate Perturbed Profiles**

**Objective:** Create new, physically realistic profiles by perturbing the representative profile using the principal components and a defined coefficient differential ($`Δc`$).

**Description:** This is the final step that synthesizes all previous work. We will take the representative profile and add a small perturbation based on one of the principal components. This perturbation is calculated by taking a principal component vector and scaling it by a "differential element" **($`±Δc_i`$)**, which we determined from the statistical analysis in Step 3 (e.g., one bin width of the histogram).

The reconstruction formula is the inverse of the PCA transformation:
$` X_{new} = \mu + C_{new} \cdot W^T `$
Where:
*   $`X_{new}`$ is the new perturbed profile.
*   $`μ`$ is the mean profile.
*   $`C_{new}`$ is the vector of new coefficients. For a perturbation along the `i`-th component, $`C_{new}`$ would be $`C_{rep} ± [0, ..., Δc_i, ..., 0]`$.
*   $`W^T`$ is the transpose of the principal components matrix.

---

### **Step 6: Generation of PCA-based Radiance Perturbation Spectra (Weighting Functions)**

**Objective:** To compute the change in downwelling spectral radiance caused by each principal mode of atmospheric variation. These radiance difference spectra, `ΔI(λ)`, will serve as our new, physically representative weighting functions.

**Description:** The core of this step is to calculate the difference between the radiance spectrum of each perturbed profile and the radiance spectrum of the unperturbed representative profile. We will have 18 such weighting functions, corresponding to the `±Δcᵢ` perturbations for each of the `k=9` principal components.

The calculation is straightforward. For each perturbation `i` (from 1 to 9) and sign (`+` or `−`), the weighting function is:

$` \Delta I_{i, \pm}(\lambda) = I_{pert, i, \pm}(\lambda) - I_{ref}(\lambda) `$

Where:
*   $`\Delta I_{i, \pm}(\lambda)`$ is the new weighting function for the i-th PC, corresponding to a coefficient change of `±Δcᵢ`. Its units are in W/(m²·sr·µm).
*   $`I_{pert, i, \pm}(\lambda)`$ is the downwelling spectral radiance computed from the atmospheric profile perturbed by `±Δcᵢ` along the i-th PC.
*   $`I_{ref}(\lambda)`$ is the spectral radiance from the unperturbed representative profile (from `Rad_NewModel66layers...280.mat`).

A critical question arises here: should we use the absolute value `|ΔI|` or retain the sign? **We must retain the sign.** The sign tells us whether a given atmospheric variation mode *increases* or *decreases* the radiance at a specific wavelength. This is fundamental physical information that is essential for designing an effective retrieval algorithm. Losing it would be like discarding half of our results.

The primary deliverable of this step will be 18 plots, each showing `ΔI(λ)` vs. wavelength `λ`, clearly identifying which PC and which sign (`+Δcᵢ` or `−Δcᵢ`) it corresponds to.

---

### **Step 7: Information Content Analysis with Δc-Normalized Weighting Functions**

**Objective:** To identify the most informative spectral regions for atmospheric sounding by analyzing a set of physically normalized, PCA-based weighting functions.

**Description:** The raw radiance difference spectra, `ΔI(λ)`, while useful, have a magnitude that depends on the chosen coefficient perturbation, `Δcᵢ`. A larger `Δcᵢ` for one PC will artificially inflate its corresponding `ΔI(λ)` relative to others. To remove this bias and analyze the pure sensitivity of radiance to atmospheric changes, we introduce a normalized weighting function.

First, we define the half-difference sensitivity, `Wᵢ(λ)`, which represents the core radiance change signal:

`Wᵢ(λ) = 0.5 · (ΔIᵢ,plus(λ) − ΔIᵢ,minus(λ)) ≈ Δcᵢ · ∂I(λ)/∂cᵢ`

Next, we normalize `Wᵢ(λ)` by its corresponding perturbation `Δcᵢ` to obtain the **unit-coefficient sensitivity**, `Uᵢ(λ)`. This crucial quantity represents the change in radiance per unit change in the `i`-th principal component coefficient and serves as our true, physically comparable weighting function.

`Uᵢ(λ) = Wᵢ(λ) / Δcᵢ ≈ ∂I(λ)/∂cᵢ`

With the set of nine `Uᵢ(λ)` spectra, we can now objectively identify the most valuable spectral "action regions." This is accomplished through a quantitative sliding-window analysis:

1.  **Sliding Window Integration:** We slide a window of width **Δλ = 0.500 μm** across the spectrum with a stride of **0.100 μm**. Within each window, we integrate each `Uᵢ(λ)` to get a vector of integrated sensitivities, `V`.
    `Vᵢ = ∫ Uᵢ(λ) dλ`

2.  **Window Metrics:** For each window, we compute metrics to quantify its information content, including the L2-norm of the sensitivity vector, `E_L2 = ||V||₂`, which represents the total signal energy in that window.

3.  **Selection of Optimal Windows:** We apply two distinct, complementary selection rules to build a portfolio of candidate spectral bands:
    *   **Top-25 by Energy:** We select the top 25 non-overlapping windows with the highest `E_L2` score. These windows represent spectral regions with the highest overall sensitivity to the principal modes of atmospheric variation. Our analysis shows these windows are primarily concentrated in the **~6.5 μm to 13.1 μm** range.
    *   **Top-5 per PC:** For each of the nine PCs, we identify the five windows that are best at isolating that specific PC's signal. This is done by maximizing a per-PC score, `scoreᵢ = E_L2 × (|Vᵢ|/Σ|Vⱼ|)`. This technique is powerful because it finds important regions even if the target PC is not the single dominant one in that window. This analysis also highlights windows primarily within the **~6.5 μm to 12.5 μm** range.

The combined results from these two selection strategies, visualized in the `overlay_U_with_top_windows_energy_and_perPC.png` plot, provide a robust, data-driven map of the most information-rich spectral bands. The strong consensus on the **~5.5 μm to 13.5 μm** region provides an excellent foundation for the subsequent filter optimization search.

---

### **Step 8: Objective Filter Set Optimization via Sequential Forward Selection**

**Objective:** To algorithmically determine the optimal set of filters (number, central wavelengths, and bandwidths) by maximizing the distinguishability of the atmospheric principal components in the measurement space.

**Description:** This step formalizes the filter selection process into a well-posed optimization problem. The goal is to select a set of filters that yields a measurement system capable of resolving the atmospheric state vector `Δc` with minimal error amplification.

The forward model links the change in the atmospheric state, represented by the vector of PCA coefficient changes `Δc`, to the resulting change in measured power at the detector, `ΔP`.

`Δ**P** = **K** ⋅ Δ**c**`

Where:
*   `Δ**P**` is the `M × 1` vector of power changes for `M` filters.
*   `Δ**c**` is the `9 × 1` vector of coefficient changes we aim to retrieve.
*   `**K**` is the `M × 9` **Jacobian matrix**. Each element `Kⱼᵢ` represents the sensitivity of filter `j` to a unit change in coefficient `cᵢ`.

Crucially, we will construct the Jacobian using our normalized weighting functions, `Uᵢ(λ)`, as they represent the true, unbiased physical sensitivity:

`Kⱼᵢ = π ∫ Uᵢ(λ) ⋅ T_filter,j(λ) ⋅ T_ZnSe(λ) ⋅ A_LiTaO₃(λ) dλ`

The quality of our filter set is determined by the mathematical properties of the Jacobian matrix `**K**`. A well-conditioned `**K**` ensures that small errors in the power measurement `Δ**P**` do not lead to large errors in the retrieved `Δ**c**`. We quantify this by the **smallest singular value** of `**K**`, denoted `σ_min`. Our optimization goal is to **maximize `σ_min`**.

We will employ a **Sequential Forward Selection (SFS)** algorithm to build the optimal filter set, exploring two different search strategies for defining the candidate filter space.

#### **Step 8.1: Define Candidate Filter Space**

We must discretize the infinite space of possible filters into a finite library.

*   **Strategy A: Local, Data-Driven Search**
    *   **Central Wavelengths (`λ_c`):** Based on the robust results from Step 7, we define a fine grid of possible central wavelengths from **5.5 μm to 13.5 μm** in steps of **0.05 μm**. This region was empirically shown to contain the vast majority of useful information.
    *   **Bandwidths (`Δλ`):** For each `λ_c`, we will test a set of three representative bandwidths: **0.2 μm, 0.5 μm, and 1.0 μm**.

*   **Strategy B: Global Search**
    *   **Central Wavelengths (`λ_c`):** We define a coarser grid across the instrument's full operational range, from **2.5 μm to 20.0 μm** in steps of **0.1 μm**.
    *   **Bandwidths (`Δλ`):** We use the same set of candidate bandwidths (0.2 μm, 0.5 μm, 1.0 μm).
    *   **Purpose:** This global search serves as a vital baseline. If its results converge on the same spectral regions identified in Strategy A, it provides powerful validation for our entire PCA-based information content analysis.

#### **Step 8.2: The Greedy Selection Algorithm**

The SFS algorithm iteratively builds the `OptimizedSet` of filters:

**Iteration 1: Select the Best First Filter**
1.  For each candidate filter `j` in the library, construct its `1 × 9` Jacobian row vector `**K**_j`.
2.  Calculate its total sensitivity, given by the squared Frobenius norm: `||**K**_j||_F² = Σᵢ(K_jᵢ)²`.
3.  Select the filter `j*` that **maximizes this norm**. This filter is the most sensitive to the combined atmospheric variability. Add it to `OptimizedSet`.

**Iteration m: Select the Best m-th Filter**
1.  With `m-1` filters already in `OptimizedSet`, iterate through all *remaining* candidate filters `l`.
2.  For each `l`, form the temporary `m × 9` Jacobian matrix `**K**_m` by appending its row vector to the Jacobian of the current `OptimizedSet`.
3.  Perform a Singular Value Decomposition (SVD) on `**K**_m` and find its smallest singular value, `σ_min`.
4.  Select the filter `l*` that **maximizes `σ_min`** and permanently add it to `OptimizedSet`. This filter is the one that adds the most new, independent information, making the atmospheric modes maximally distinguishable.

#### **Step 8.3: Determine the Optimal Number of Filters (Stopping Criterion)**

We continue the iterative selection process, plotting the maximized `σ_min` at each step versus the number of filters (`m`). The process stops when:
*   We reach the "knee" of the curve, where adding more filters yields diminishing returns in `σ_min`. This indicates we have captured the bulk of the available information.
*   We have selected at least `M=9` filters, the theoretical minimum required to solve for `k=9` unknown coefficients. Extending to `M=11` or `M=12` can add valuable redundancy and robustness against noise.

The final `OptimizedSet` will contain the central wavelengths and bandwidths for the objectively determined optimal filter configuration, with results from both the local and global search strategies for comparison.