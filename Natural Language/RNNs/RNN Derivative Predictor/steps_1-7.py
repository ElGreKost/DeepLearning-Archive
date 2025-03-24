# # Βήμα 2: Data parsing

# %%
import os

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def dataparser(folder_path):
    digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    digit_set = set(digits)  # Για ταχύτερο έλεγχο περιέχοντος

    wav_list, speaker_list, digit_list = (
        [],
        [],
        [],
    )  # FIXME: Από wav μετονομάστηκε σε wav_list (για να ταιριάζει με το return)

    for filename in filter(lambda f: f.endswith(".wav"), os.listdir(folder_path)):
        name, ext = os.path.splitext(filename)
        for digit in digit_set:
            if name.startswith(digit):
                try:
                    speaker = int(name[len(digit) :])
                    temp_wav, _ = librosa.load(
                        os.path.join(folder_path, filename), sr=16000
                    )
                    wav_list.append(temp_wav)
                    speaker_list.append(speaker)
                    digit_list.append(digit)
                except:
                    pass
                break  # βγαίνει από τον βρόγχο μόλις βρει το σωστό ψηφίο

    return wav_list, speaker_list, digit_list


# ## Visualize Results

wav_list, speaker_list, digit_list = dataparser("data/digits")

sample_rate = 16000
sample_wav = wav_list[0]

plt.figure(figsize=(12, 4))
plt.title(f"Sample waveform for digit: {digit_list[0]}, Speaker: {speaker_list[0]}")
librosa.display.waveshow(sample_wav, sr=sample_rate)
plt.show()

ipd.Audio(sample_wav, rate=sample_rate)


# # Βήμα 3 MFCC


def extract_mfcc_features(
    wav_list, sr=16000, n_mfcc=13, window_length_ms=25, hop_length_ms=10
):
    n_fft = int(sr * window_length_ms / 1000)  # length of fft window.
    hop_length = int(
        sr * hop_length_ms / 1000
    )  # number of samples between successive frames.

    mfcc_features = [
        np.vstack(
            (
                mfcc := librosa.feature.mfcc(
                    y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
                ),
                librosa.feature.delta(mfcc),
                librosa.feature.delta(mfcc, order=2),
            )
        )
        for signal in wav_list
    ]

    return mfcc_features


wav_list, speaker_list, digit_list = dataparser("data/digits")
mfcc_features = extract_mfcc_features(wav_list)

print("MFCC Features Shape:", [mfcc.shape for mfcc in mfcc_features])


# # Βήμα 4

import matplotlib.pyplot as plt


def plot_mfcc_histograms(mfcc_features, digit_list, digits, n_mfccs=[0, 1], bins=30):
    plt.figure(figsize=(15, len(n_mfccs) * 3))
    for i, n in enumerate(n_mfccs):
        for j, digit in enumerate(digits):
            # Extract the nth column of each mfcc 2D array where digit matches
            mfccs = [
                mfcc[n] for mfcc, d in zip(mfcc_features, digit_list) if d == digit
            ]
            # Flatten the list of arrays
            mfccs_flat = [item for sublist in mfccs for item in sublist]
            # Calculate the standard deviation
            std_dev = np.std(mfccs_flat)
            # Plot the histogram for the nth MFCC of the specified digit
            plt.subplot(len(n_mfccs), len(digits), i * len(digits) + j + 1)
            plt.hist(mfccs_flat, bins=bins, alpha=0.7, label=f"Std Dev: {std_dev:.2f}")
            plt.legend(loc="upper right")
            plt.title(f"Histogram of {n + 1}st MFCC for {digit}")
    plt.tight_layout()
    plt.show()


wav_list, speaker_list, digit_list = dataparser("data/digits")
mfcc_features = extract_mfcc_features(wav_list)

digits = ["one", "two", "three", "four"]
n_mfccs = [0, 1, 2]

plot_mfcc_histograms(mfcc_features, digit_list, digits, n_mfccs)


# Παράδειγμα χρήσης με την data_parser
folder_path = "data/digits"
wav, speaker_list, digit_list = dataparser(folder_path)

# Ψηφία για απεικόνιση
n1, n2 = "six", "nine"
digits = [n1, n2]
n_mfccs = [0, 1]  # 1ος και 2ος MFCC (0-based index)

# Εξαγωγή MFCC χαρακτηριστικών
mfcc_features = extract_mfcc_features(wav)

# Δημιουργία ιστογραμμάτων για τα MFCCs
plot_mfcc_histograms(mfcc_features, digit_list, digits, n_mfccs)


# Modified MFSC extraction function without averaging
def extract_mfsc_features(
    wav_list, sr=16000, n_mfsc=13, window_length_ms=25, hop_length_ms=10
):
    n_fft = int(sr * window_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)
    mfsc_features = [
        librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mfsc
        )
        for signal in wav_list
    ]
    return mfsc_features


# Use imshow to plot each feature as a 2D array
def plot_features_comparison(feature_list, title):
    plt.figure(figsize=(12, 6))
    for i, features in enumerate(feature_list):
        plt.subplot(1, len(feature_list), i + 1)
        plt.imshow(features, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar()
        plt.title(f"{title} {i + 1}")
    plt.tight_layout()
    plt.show()


# Autocorrelation function that calculates for each column
def autocorrelation(signal):
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)

    # Normalize the signal
    signal = signal - mean
    autocorr = np.correlate(signal, signal, mode="full")[n - 1 :] / (var * n)
    return autocorr


# Show correlation for each 2D coefficient matrix in the list
def show_correlation(coef_list):
    mean_autocorr = []

    # Iterate over each 2D array in coef_list
    for signal in coef_list:
        # Compute the autocorrelation for each column in the 2D array
        col_autocorrs = [
            np.mean(autocorrelation(signal[:, i])) for i in range(signal.shape[1])
        ]
        # Append the mean autocorrelation of all columns
        mean_autocorr.append(np.mean(col_autocorrs))

    return np.array(mean_autocorr)


# Example usage of feature extraction and correlation
folder_path = "data/digits"
wav, speaker_list, digit_list = dataparser(folder_path)

selected_signals = [wav[i] for i, d in enumerate(digit_list) if d in ["six", "nine"]][
    :2
]

# Extract MFSC and MFCC features
mfsc_features = extract_mfsc_features(selected_signals)
selected_mfcc_features = extract_mfcc_features(selected_signals)

# Plot MFSC and MFCC comparisons
plot_features_comparison(mfsc_features, "MFSC")
plot_features_comparison([mfcc[:13] for mfcc in selected_mfcc_features], "MFCC")

# Compute and print mean autocorrelation for both feature sets
mfsc_mean_autocorr = show_correlation(mfsc_features)
mfcc_mean_autocorr = show_correlation(selected_mfcc_features)

print("Mean Autocorrelation for MFSC Features:", mfsc_mean_autocorr)
print("Mean Autocorrelation for MFCC Features:", mfcc_mean_autocorr)


# # Βήμα 5

import matplotlib.pyplot as plt
import numpy as np


def create_unique_vector(matrix):
    unique_vector = []
    for row in matrix:
        row_mean = np.mean(row)
        row_std = np.std(row)
        unique_vector.extend([row_mean, row_std])
    return np.array(unique_vector)


# Generate unique vectors for each set of features
mfcc_features_unique = [create_unique_vector(features) for features in mfcc_features]


def create_scatterplot(mfcc_features_unique, digit_list):
    # Extract the first two elements from each unique vector
    x_vals = [vec[0] for vec in mfcc_features_unique]
    y_vals = [vec[1] for vec in mfcc_features_unique]

    # Get unique digits and assign a color to each
    unique_digits = list(set(digit_list))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_digits)))
    color_map = {digit: color for digit, color in zip(unique_digits, colors)}

    plt.figure(figsize=(10, 8))

    # Plot each point with the color corresponding to its label
    for x, y, digit in zip(x_vals, y_vals, digit_list):
        plt.scatter(x, y, color=color_map[digit], label=digit)

    # Add legend for each unique digit label
    handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", label=str(digit))
        for digit, color in color_map.items()
    ]
    plt.legend(handles=handles, title="Digits")

    plt.xlabel("Mean of 1st MFCC coefficient")
    plt.ylabel("Standard Deviation of 1st MFCC coefficient")
    plt.title("Scatter Plot of Mean and Standard Deviation of 1st MFCC coefficient")
    plt.grid(True)
    plt.show()


# Example usage:
create_scatterplot(mfcc_features_unique, digit_list)


# # Βήμα 6


# PCA - Choose number of dimensions
def reduce_pca(data_list, n_dim):
    # Stack the list of 1D arrays into a 2D array for PCA
    data_matrix = np.vstack(data_list)

    # Perform PCA
    pca = PCA(n_components=n_dim)
    reduced_matrix = pca.fit_transform(data_matrix)

    # Calculate variance explained by the reduced dimensions
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(
        f"Percentage of variance preserved with {n_dim} dimensions: {explained_variance:.2f}%"
    )

    # Convert the reduced 2D array back into the original format (list of 1D arrays)
    reduced_data_list = [reduced_matrix[i] for i in range(reduced_matrix.shape[0])]
    return reduced_data_list


# Reduce mfcc_features_unique to 2D and 3D
mfcc_features_unique_2dim = reduce_pca(mfcc_features_unique, n_dim=2)
mfcc_features_unique_3dim = reduce_pca(mfcc_features_unique, n_dim=3)

# Plot the 2D representation
create_scatterplot(mfcc_features_unique_2dim, digit_list)


def create_3d_scatterplot(data, digit_list):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_digits = list(set(digit_list))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_digits)))
    color_map = {digit: color for digit, color in zip(unique_digits, colors)}

    # Plot each point with the color corresponding to its label
    for vec, digit in zip(data, digit_list):
        ax.scatter(vec[0], vec[1], vec[2], color=color_map[digit], label=digit)

    handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", label=str(digit))
        for digit, color in color_map.items()
    ]
    ax.legend(handles=handles, title="Digits")

    ax.set_xlabel("1st Principal Component")
    ax.set_ylabel("2nd Principal Component")
    ax.set_zlabel("3rd Principal Component")
    ax.set_title("3D Scatter Plot of PCA-reduced MFCC Features")
    plt.show()


# Example usage for 3D scatterplot:
create_3d_scatterplot(mfcc_features_unique_3dim, digit_list)


# # Βήμα 7


# Step 1: PCA to reduce dimensions while preserving 95% of variance
def pca_reduction(data_list, variance_threshold=0.95):
    data_matrix = np.vstack(data_list)  # Stack into a 2D array for PCA
    pca = PCA(
        n_components=variance_threshold, svd_solver="full"
    )  # Full solver to keep explained variance
    data_reduced = pca.fit_transform(data_matrix)

    # Print the explained variance for the selected components
    print(
        f"Explained variance ratio with {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.2%}"
    )

    # Return the reduced data in the original list format
    return [data_reduced[i] for i in range(len(data_list))]


# Reduce mfcc_features to preserve 95% of the variance
mfcc_features_unique_95 = pca_reduction(mfcc_features_unique)

# Step 2: Prepare data for training and testing
# Convert labels (digit_list) to a numpy array for indexing
X = np.array(mfcc_features_unique_95)
y = np.array(digit_list)

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train and test classifiers
# Define and train classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

# Train, predict, and calculate accuracy for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.2%}")


# Step 1: PCA to reduce dimensions while preserving 95% of variance
def pca_reduction(data_list, variance_threshold=0.95):
    data_matrix = np.vstack(data_list)  # Stack into a 2D array for PCA
    pca = PCA(
        n_components=variance_threshold, svd_solver="full"
    )  # Full solver to keep explained variance
    data_reduced = pca.fit_transform(data_matrix)

    # Print the explained variance for the selected components
    print(
        f"Explained variance ratio with {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.2%}"
    )

    # Return the reduced data in the original list format
    return [data_reduced[i] for i in range(len(data_list))]


# Reduce mfcc_features to preserve 95% of the variance
mfcc_features_unique_95 = pca_reduction(mfcc_features_unique)

# Step 2: Normalize the data
scaler = StandardScaler()
mfcc_features_unique_95_normalized = scaler.fit_transform(
    np.array(mfcc_features_unique_95)
)

# Convert labels (digit_list) to a numpy array for indexing
X = mfcc_features_unique_95_normalized
y = np.array(digit_list)

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train and test classifiers
# Define and train classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

# Train, predict, and calculate accuracy for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.2%}")
