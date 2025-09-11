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

### **Step 7: Information Content Analysis for Optimal Filter Placement**

**Objective:** To identify the most informative spectral regions for atmospheric sounding by analyzing the set of new weighting functions.

**Description:** Not all wavelengths are created equal. Some are highly sensitive to atmospheric changes, while others are not. We will identify the "action regions" by examining our 18 `ΔI(λ)` spectra.

The methodology is as follows:
1.  **Overlay the Weighting Functions:** Plot all 18 $`ΔI_{i, \pm}(\lambda)`$ spectra on a single graph. This will create a visual map of atmospheric sensitivity.
2.  **Identify High-Signal Regions:** Look for spectral bands where the `ΔI` values (both positive and negative) have the largest magnitudes. These are the wavelengths where the principal modes of atmospheric variability produce the strongest radiance changes. These are prime candidates for filter placement.
3.  **Identify Regions of Orthogonality:** Look for bands where the spectral signatures of different PCs are distinct. For example, a region where `ΔI₁` is strongly positive while `ΔI₂` is strongly negative is extremely valuable, as a filter placed there can help differentiate between these two atmospheric modes.

This qualitative analysis will guide our initial filter selection, pointing us toward the spectral windows that contain the most information about the atmospheric state as defined by our PCA basis.

---

### **Step 8: Quantifying Detectable Signals for a Realistic Instrument**

**Objective:** To connect the theoretical radiance changes to the practical capabilities of your detector system and determine if the signals are strong enough to be reliably measured.

**Description:** A large radiance change is useless if it's smaller than your instrument's noise floor. In this step, we will calculate the actual change in power that your detector would register for each perturbation and compare it to the detector's noise level.

For a given hypothetical optical filter with transmittance `T_filter(λ)`, the change in detected power `ΔP` for a given perturbation is calculated by integrating the radiance change through your complete detector system.

$` \Delta P_{i, \pm} = \pi \cdot \int \Delta I_{i, \pm}(\lambda) \cdot T_{filter}(\lambda) \cdot T_{ZnSe}(\lambda) \cdot A_{LiTaO_3}(\lambda) \,d\lambda `$

Where:
*   $`\Delta P_{i, \pm}`$ is the change in absorbed power (in Watts) at the detector for the perturbation `i,±`.
*   The factor of `π` is included to convert from spectral radiance (W/m²·sr·µm) to spectral irradiance (W/m²·µm) assuming isotropic downwelling radiance.
*   $`T_{ZnSe}(\lambda)`$ is the transmittance of the Zinc Selenide window.
*   $`A_{LiTaO_3}(\lambda)`$ is the spectral absorptivity of the Lithium Tantalate detector material.

We then compare this signal to the detector's noise. The noise equivalent power (NEP) is given as 22.6 nW/√Hz. The total noise power `P_noise` depends on the measurement bandwidth `B` (in Hz), which is inversely related to the integration time `τ` (i.e., `B ≈ 1/τ`).

$` P_{noise} = NEP \cdot \sqrt{B} `$

A signal `ΔP` is considered detectable if its magnitude is significantly greater than `P_noise`. This analysis allows us to refine the filter locations identified in Step 7 by ensuring that the resulting power changes are measurable by your actual hardware.

---

### **Step 9 (Revised): Objective Filter Set Optimization via Sequential Forward Selection**

**Objective:** To algorithmically determine the optimal number, central wavelengths, and bandwidths of the filters by maximizing the information content and minimizing the retrieval error of the atmospheric state.

**Description:** The core idea is to treat the filter selection as a formal optimization problem. We want to select a set of filters that makes the different principal modes of the atmosphere as distinguishable as possible in the measurement space (i.e., the space of detector power readings).

First, let's formalize our forward model. The change in the atmospheric state can be described by the vector of PCA coefficient changes, `Δc`.

$` \Delta\mathbf{c} = [\Delta c_1, \Delta c_2, ..., \Delta c_9]^T `$

The change in power measured by a filter `j`, `ΔPⱼ`, is a linear function of these coefficient changes:

$` \Delta P_j = \sum_{i=1}^{9} K_{ji} \cdot \Delta c_i `$

This can be written in matrix form:

$` \Delta\mathbf{P} = \mathbf{K} \cdot \Delta\mathbf{c} `$

Where:
*   **$`ΔP`$** is an `M × 1` vector of power changes for `M` filters.
*   **$`Δc`$** is the `9 × 1` vector of coefficient changes we want to retrieve.
*   **$`K`$** is the `M × 9` **Jacobian matrix** (or kernel matrix). Each element $`K_{ji}`$ represents the sensitivity of filter `j` to a change in coefficient `cᵢ`. It is calculated from our previously derived weighting functions:

    $` K_{ji} = \frac{\pi}{\Delta c_i} \int \Delta I_{i,+}(\lambda) \cdot T_{filter,j}(\lambda) \cdot T_{ZnSe}(\lambda) \cdot A_{LiTaO_3}(\lambda) \,d\lambda `$
    (Note: We use the `+Δcᵢ` perturbation by convention, assuming linearity holds for `−Δcᵢ` as well).

The goal of the inverse problem is to find **$`Δc`$** given a measurement of **$`ΔP`$**. A good filter set will produce a Jacobian matrix **$`K`$** that is well-conditioned, meaning the inversion is stable and does not amplify measurement noise. The "goodness" of **$`K`$** can be quantified objectively. A powerful metric is the **smallest singular value** of **$`K`$**, denoted $`σ_min`$. Maximizing $`σ_min`$ is equivalent to minimizing the worst-case error amplification.

We will use a **Sequential Forward Selection (SFS)** algorithm, a greedy approach that builds the optimal filter set one filter at a time.

#### **Step 9.1: Define the Candidate Filter Space**

We cannot search a continuous space of all possible filters. Instead, we must discretize it.
1.  **Central Wavelengths (`λ_c`):** Define a grid of possible central wavelengths. For example, from 4 µm to 20 µm in steps of 0.1 µm. This grid should cover all regions where you observed significant radiance changes in Step 7.
2.  **Bandwidths (`Δλ`):** For each central wavelength, define a few possible bandwidths (e.g., 0.2 µm, 0.5 µm, 1.0 µm). Assume a simple boxcar or Gaussian shape for the filter transmittance `T_filter(λ)`.

This creates a finite library of several hundred or a few thousand "candidate filters."

#### **Step 9.2: The Greedy Selection Algorithm**

The algorithm proceeds as follows:

**Iteration 1: Select the Best First Filter**
1.  For each candidate filter `j` in your library, construct the `1 × 9` Jacobian matrix **$`K_j`$**.
2.  Calculate a metric of its total sensitivity. A good metric is the squared Frobenius norm, which is the sum of squares of its elements: **$`||K_j||_F² = Σᵢ(K_{ji})²`$**.
3.  Select the filter `j*` that **maximizes this norm**. This filter is the one most sensitive to the dominant modes of atmospheric variability. Add it to your `OptimizedSet`.

**Iteration 2: Select the Best Second Filter**
1.  Now, for every *remaining* candidate filter `k` in your library, temporarily add it to your `OptimizedSet`. This creates a `2 × 9` Jacobian matrix **$`K_{j*, k}`$**.
2.  Perform a Singular Value Decomposition (SVD) on this **$`K_{j*, k}`$** matrix and find its smallest singular value, $`σ_min`$.
3.  Select the filter $`k*`$ that **maximizes $`σ_min`$**. This filter is the one that adds the most *new, independent* information to the system, making it easier to distinguish atmospheric modes that might have looked similar to the first filter. Add $`k*`$ to your `OptimizedSet`.

**Iteration m: Select the Best m-th Filter**
1.  With `m-1` filters already in `OptimizedSet`, iterate through all remaining candidate filters `l`.
2.  For each, form the `m × 9` Jacobian matrix **$`K_m`$** and find its smallest singular value, $`σ_min`$.
3.  Select the filter $`l*`$ that maximizes $`σ_min`$ and add it to `OptimizedSet`.

#### **Step 9.3: Determine the Optimal Number of Filters (Stopping Criterion)**

Continue the iterative process. With each new filter added, plot the value of the maximized $`σ_min`$ versus the number of filters (`m`). You will typically see a curve that rises sharply at first and then begins to flatten out.

You can stop when:
*   You reach the "knee" of the curve, where adding another filter provides only a marginal improvement in $`σ_min`$. This indicates you have captured most of the available information.
*   You have at least `M=9` filters, which is the theoretical minimum to resolve `k=9` unknown coefficients. Continuing to `M=11` or `M=12` can add robustness and redundancy.

The final `OptimizedSet` contains the central wavelengths and bandwidths of your objectively determined optimal filter set.