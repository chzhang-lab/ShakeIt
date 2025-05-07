#!/usr/bin/env python

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import os
import sys
import getopt

def usage():
    print("""
ShakeIt — A tool for analyzing ligand conformational distributions from molecular dynamics (MD) simulations using Gaussian mixture models (GMMs).

Usage:
    python shakeit.py -i <input_file> [-o <output_file>] [-m <sg|all|norm>] [-r <column>] 
                      [-s <skip_lines>] [-p <show|path_to_file>] [-t <gmm1|gmm2>]
Options:
    -i, --input         Required. Path to the input file (e.g., rmsd.dat).
Optional Arguments:
    -o, --output        Output file name for results (e.g., output.dat).
    -m, --output_mode   Specifies the type of output to generate:
        sg              (default) Outputs the Sg score summarizing conformational spread.
        all             Outputs all parameters (weights, means, standard deviations) 
                        of each Gaussian component.
        norm            Outputs normalized mean and standard deviation values.
    -r, --column        Column index to read from the input file (default: 2).
    -s, --skip          Number of initial lines to skip in the input file (default: 0).
    -p, --plot          Controls plot generation:
        show            (default) Displays the plot interactively.
        <path_to_file>  Saves the plot to the specified file.
    -t, --method        GMM fitting method:
        gmm1            (default) Non-Bayesian fitting using a Trust-Region Reflective
                        nonlinear least-squares solver (SciPy).
        gmm2            Bayesian fitting using the Expectation-Maximization algorithm 
                        (scikit-learn implementation).
    -h, --help          Displays this help message and exits.
""")

def parse_args():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], 
            "hi:o:m:r:s:p:t:", 
            ["help", "input=", "output=","output_mode=","column=", "skip=", "plot=", "method="]
        )
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        usage()
        sys.exit(2)

    input_file = None
    output_file = None
    output_mode = "sg"
    column = 2
    skip = 0
    plot = None
    method = "gmm1"

    for name, value in opts:
        if name in ("-h", "--help"):
            usage()
            sys.exit()
        if name in ("-i", "--input"):
            input_file = value
        if name in ("-o", "--output"):
            output_file = value
        if name in ("-m", "--output_mode"):
            if value not in ["all", "sg", "norm"]:
                print("Error: Output mode must be all, sg or norm.")
                usage()
                sys.exit(2)
            output_mode = value
        if name in ("-r", "--column"):
            column = int(value)
        if name in ("-s", "--skip"):
            skip = int(value)
        if name in ("-p", "--plot"):
            plot = value
        if name in ("-t", "--method"):
            if value not in ["gmm1", "gmm2"]:
                print("Error: Method must be gmm1 or gmm2.")
                usage()
                sys.exit(2)
            method = value

    if not input_file:
        print("Error: Missing required argument -i (input file).")
        usage()
        sys.exit(1)

    return input_file, output_mode, output_file, column, skip, plot, method

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        height = params[i]
        mean = params[i + 1]
        width = params[i + 2]
        y += height * np.exp(-((x - mean) ** 2) / (2 * width ** 2))
    return y

def fit_gmm1(guess, func, x, y, lower_bounds, upper_bounds):
    try:
        popt, pcov = curve_fit(func, x, y, p0=guess, bounds=(lower_bounds, upper_bounds), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return popt.reshape(-1, 3), True, pcov, perr.reshape(-1, 3)
    except Exception as e:
        print("Gaussian fitting failed:", e)
        return [], False, None, None

def fit_gmm2(data, max_components=8):
    x = data.reshape(-1, 1)
    n_components_range = range(1, max_components+1)
    bics = []
    gmms = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, max_iter=100, random_state=1)
        gmm.fit(x)
        bics.append(gmm.bic(x))
        gmms.append(gmm)

    min_bic = min(bics)
    threshold = min_bic + 0.03 * abs(min_bic)

    candidates = [
        {"n": n, "bic": bic, "idx": i}
        for i, (n, bic) in enumerate(zip(n_components_range, bics)) if bic <= threshold
    ]

    if not candidates:
        best_i = np.argmin(bics)
        candidates = [{"n": n_components_range[best_i], "bic": bics[best_i], "idx": best_i}]

    best_candidate = min(candidates, key=lambda c: c["n"])
    best_n = best_candidate["n"]
    best_gmm = gmms[best_candidate["idx"]]

    for trial in range(3):
        gmm = GaussianMixture(n_components=best_n, max_iter=100, 
                             init_params='k-means++', random_state=trial)
        gmm.fit(x)
        if gmm.bic(x) < best_gmm.bic(x):
            best_gmm = gmm

    weights   = np.round(best_gmm.weights_.flatten(), 4)
    means     = np.round(best_gmm.means_.flatten(), 4)
    std_devs  = np.round(np.sqrt(best_gmm.covariances_).flatten(), 4)
    return weights, means, std_devs

def weighting_parm(sigma_values, mean_values, height_values, method):
    sigma_weight = 0
    mean_weight = 0
    if method == "gmm1":
        s_values = [h * sigma for h, sigma in zip(height_values, sigma_values)]
        total_s = sum(s_values)
        weights = [round(s / total_s, 4) for s in s_values]
        for sigma, mean, weight in zip(sigma_values, mean_values, weights):
            sigma_weight += sigma * weight
            mean_weight += mean * weight
    elif method == "gmm2":
        for sigma, mean, weight in zip(sigma_values, mean_values, height_values):
            sigma_weight += sigma * weight
            mean_weight += mean * weight
            weights = height_values
    return sigma_weight, mean_weight, weights

def sg_calculate(n, mean_weight, sigma_weight, a, b):
    sg = a*mean_weight + (1-a)*(b*sigma_weight*n + (1-b)*np.log(n+4))
    return sg

def main(input_file, output_mode, output_file, column, skip, plot, method):
    data_raw = np.loadtxt(input_file, skiprows=skip)
    data = data_raw[:, column - 1]

    if np.mean(data) > 1000:
        print("Error in MD")
        sys.exit(1)

    if method == "gmm1":
        hist, bins = np.histogram(data, bins=100, density=True)
        bin_edges = bins[:-1]
        max_v = max(data)
        min_v = min(data)
        width = (max_v - min_v) / 100
        x_hist = np.around(bin_edges + (width / 2), decimals=2)
        y = hist

        Y = savgol_filter(y, 11, 3, mode='nearest')
        y_fit = savgol_filter(Y, 11, 3, mode='nearest')

        ori_peaks = scipy.signal.find_peaks_cwt(y_fit, [3])
        filtered_valleys = scipy.signal.find_peaks_cwt(-y_fit, [3])

        peaks = []
        prev_peak = ori_peaks[0]
        threshold = 0.004 / width

        for peak in ori_peaks[1:]:
            if peak - prev_peak > 4 and any((valley > prev_peak) and (valley < peak) for valley in filtered_valleys):
                peaks.append(prev_peak)
                prev_peak = peak
            elif y_fit[peak] >= y_fit[prev_peak]:
                prev_peak = peak

        peaks.append(prev_peak)
        peaks = [peak for peak in peaks if y_fit[peak] > threshold]

        distances = []
        for peak in peaks:
            closest_valley = min(filtered_valleys, key=lambda x: abs(x - peak))
            if abs(peak - closest_valley) < 4:
                filtered_valleys_copy = [valley for valley in filtered_valleys if abs(valley - peak) >= 4]
                closest_valley = min(filtered_valleys_copy, key=lambda x: abs(x - peak))
            if abs(y_fit[peak] - y_fit[closest_valley]) < 0.03:
                filtered_valleys_copy = [valley for valley in filtered_valleys if abs(y_fit[valley] - y_fit[peak]) >= 0.03]
                closest_valley = min(filtered_valleys_copy, key=lambda x: abs(x - peak))

            distance = abs(x_hist[peak] - x_hist[closest_valley])
            rounded_distance = round(distance, 2)
            distances.append(rounded_distance)

        peak_indexes_xs_ys = np.asarray([list(a) for a in zip(y_fit[peaks], x_hist[peaks])])
        n = len(peaks)
        combined_data = np.column_stack((peak_indexes_xs_ys[:, 0], peak_indexes_xs_ys[:, 1], distances))
        get_highest_n_peaks_scipy = sorted(combined_data, key=lambda pair: pair[0], reverse=True)[-n:]
        n_peaks = np.asarray([[np.around(a[0], decimals=2), np.around(a[1], decimals=2), np.around(a[2], decimals=2)] for a in get_highest_n_peaks_scipy])

        scaled_min_list = []
        scaled_max_list = []
        guess = []
        for idx, xs_ys in enumerate(n_peaks):
            h = xs_ys[0]
            guess.append(h)
            scaled_min_list.append(h * 0.8)
            scaled_max_list.append(h * 2)
            m = xs_ys[1]
            guess.append(m)
            scaled_min_list.append(m * 0.7)
            scaled_max_list.append(m * 2)
            s = xs_ys[2]
            guess.append(s)
            scaled_min_list.append(s * 0.2)
            scaled_max_list.append(s * 3)

        params, success, pcov, perr = fit_gmm1(guess, func, x_hist, y, scaled_min_list, scaled_max_list)

        if not success:
            return

        height_list, mean_list, sigma = [], [], []
        for index, param in enumerate(params):
            height = param[0]
            mean = param[1]
            width = abs(param[2])
            if width == 0 or height <= 0:
                continue
            if height > guess[index * 3] * 100 or height <= guess[index * 3] / 2.5:
                continue
            if width > guess[index * 3 + 2] * 3.2:
                continue
            height_list.append(np.round(height, 4))
            mean_list.append(np.round(mean, 4))
            sigma.append(np.round(width, 4))

    elif method == "gmm2":
        weights, means, std_devs = fit_gmm2(data)
        height_list = weights.tolist()
        mean_list = means.tolist()
        sigma = std_devs.tolist()

    N = len(sigma)
    sigma_weight, mean_weight, weights = weighting_parm(sigma, mean_list, height_list, method)
    a = 0.88
    b = 0.38
    sg = sg_calculate(N, mean_weight, sigma_weight, a, b)

    if output_file:
        with open(output_file, 'w') as f:
            if output_mode == "sg":
                f.write(f"sg={sg:.4f}\n")
            elif output_mode == "norm":
                f.write(f"mean_normalized={mean_weight:.4f}\nsigma_normalized={sigma_weight:.4f}\nn={N}\nsg={sg:.4f}\n")
            elif output_mode == "all":
                for idx, (h, m, s) in enumerate(zip(weights, mean_list, sigma), 1):
                    f.write(f"sigma{idx}={s:.4f}\tmean{idx}={m:.4f}\tweight{idx}={h:.4f}\n")
                f.write(f"n={N}\nsg={sg:.4f}\n")
        print(f"Results saved to {output_file}")
    else:
        if output_mode == "sg":
            print(f"sg={sg:.4f}")
        elif output_mode == "norm":
            print(f"mean_normalized={mean_weight:.4f}\nsigma_normalized={sigma_weight:.4f}\nn={N}\nsg={sg:.4f}")
        elif output_mode == "all":
            for idx, (h, m, s) in enumerate(zip(weights, mean_list, sigma), 1):
                print(f"sigma{idx}={s:.4f}\tmean{idx}={m:.4f}\tweight{idx}={h:.4f}")
            print(f"n={N}\nsg={sg:.4f}")

    if plot:
        plt.figure(figsize=(10, 6))
        
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0], 
                alpha=0.5, label='Distribution')

        x_plot = np.linspace(data.min(), data.max(), 1000)
        total_pdf = np.zeros_like(x_plot)

        for i in range(len(weights)):
            comp_pdf = weights[i] * norm.pdf(x_plot, loc=mean_list[i], scale=sigma[i])
            plt.plot(x_plot, comp_pdf, '--', label=f'g{i+1}(x), w{i+1}={weights[i]}, μ{i+1}={mean_list[i]}, σ{i+1}={sigma[i]}')
            total_pdf += comp_pdf

        plt.plot(x_plot, total_pdf, '-', linewidth=2, color='Gray', label='G(x)')
        plt.title(f'{os.path.splitext(os.path.basename(input_file))[0]} ({method.upper()} Fit)')
        plt.xlabel('RMSD(Å)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        if plot == "show":
            plt.show()
        else:
            plt.savefig(plot, dpi=300)
            print(f"Plot saved to {plot}")

if __name__ == "__main__":
    args = parse_args()
    main(*args)
