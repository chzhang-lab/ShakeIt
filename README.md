## ShakeIt

A command‑line tool for analyzing ligand conformational distributions from molecular dynamics (MD) simulations using Gaussian mixture models (GMMs).

---

## Features

* Fits MD-derived RMSD or other coordinate distributions to Gaussian mixture models (GMMs) via two algorithms:

  * **gmm1**: Non‑Bayesian Trust‑Region Reflective nonlinear least‑squares fitting (SciPy).
  * **gmm2**: Bayesian Expectation‑Maximization fitting (scikit‑learn).
* Calculates an Sg score summarizing the binding stability.
* Outputs either the Sg score, normalized parameters, or all component parameters.
* Generates plots of the distribution and GMM components.

---

## Requirements

* Python 3.7 or later
* NumPy
* SciPy
* scikit-learn
* Matplotlib

Install dependencies via pip:

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/ShakeIt.git
cd ShakeIt
```

Make sure `shakeit.py` is executable:

```bash
chmod +x shakeit.py
```

---

## Usage

```bash
python shakeit.py -i <input_file> [options]
```

### Required Argument

* `-i, --input` \<input\_file>

  * Path to a ligand rmsd data file (e.g., `rmsd.dat`).

### Optional Arguments

| Flag                | Description                                                                | Default |
| ------------------- | -------------------------------------------------------------------------- | ------- |
| `-o, --output`      | Write results to this file (e.g., `results.dat`).                          | stdout  |
| `-m, --output_mode` | `sg` — Sg score; `norm` — normalized mean & σ; `all` — all GMM parameters. | `sg`    |
| `-r, --column`      | Column index (1‑based) to read from input.                                 | `2`     |
| `-s, --skip`        | Number of header lines to skip in input file.                              | `0`     |
| `-p, --plot`        | `show` — display plot; `<path>` — save plot to file.                       | none    |
| `-t, --method`      | `gmm1` — SciPy nonlinear least‑squares; `gmm2` — scikit‑learn EM.          | `gmm2`  |
| `-h, --help`        | Show help message and exit.                                                |         |

### Examples

1. **Compute Sg score** from `rmsd.dat` and display plot:

   ```bash
   python shakeit.py -i rmsd.dat -p show
   ```

2. **Save all GMM parameters** (w, μ, σ, n, sg) to `output.txt`:

   ```bash
   python shakeit.py -i rmsd.dat -o output.txt -m all
   ```

3. **Fit with Bayesian EM** and save normalized values:

   ```bash
   python shakeit.py -i rmsd.dat -m norm -t gmm2 -o output.txt
   ```

4. **Skip header** lines in input and save plot to file:

   ```bash
   python shakeit.py -i rmsd.dat -s 100 -p plot.png
   ```

---

## Output Formats

* **sg**: single-line `sg=VALUE`
* **norm**: multi-line:

  ```text
  mean_normalized=...
  sigma_normalized=...
  n=...
  sg=...
  ```
* **all**: each Gaussian component on its own line, then `n` and `sg`.

---

## License

MIT License
