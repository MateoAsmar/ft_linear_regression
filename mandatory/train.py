import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"
THETAS_FILE = BASE_DIR / "thetas.csv"


def read_data(filename):
    mileage, price = [], []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Dataset is empty.") from None
        if len(header) < 2:
            raise ValueError("CSV must contain mileage and price columns.")

        for line_number, row in enumerate(reader, start=2):
            try:
                x, y = float(row[0]), float(row[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid mileage or price on CSV line {line_number}.") from None
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Mileage and price must be finite on CSV line {line_number}.")
            mileage.append(x)
            price.append(y)

    if not mileage:
        raise ValueError("Dataset is empty.")

    return mileage, price


def normalize(data):
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return [0.0 for x in data], min_val, max_val

    normalized = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized, min_val, max_val


def denormalize_theta(theta0, theta1, min_m, max_m, min_p, max_p):

    scale_m = max_m - min_m
    scale_p = max_p - min_p

    if scale_m == 0:
        raise ValueError("Mileage values must not all be identical.")

    real_theta1 = theta1 * (scale_p / scale_m)
    real_theta0 = min_p + (scale_p * theta0) - (real_theta1 * min_m)

    return real_theta0, real_theta1


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def train_model(mileage, price, learning_rate=0.01,
                iterations=10000):

    theta0 = 0.0
    theta1 = 0.0
    m = len(mileage)

    for _ in range(iterations):
        sum_error0 = 0.0
        sum_error1 = 0.0

        for i in range(m):
            prediction = estimate_price(mileage[i], theta0, theta1)
            error = prediction - price[i]

            sum_error0 += error
            sum_error1 += error * mileage[i]

        tmp0 = learning_rate * sum_error0 / m
        tmp1 = learning_rate * sum_error1 / m

        theta0 -= tmp0
        theta1 -= tmp1

    return theta0, theta1


def save_thetas(theta0, theta1, filename):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])


def main():
    try:
        mileage, price = read_data(DATA_FILE)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Cannot read dataset: {error}")
        return

    norm_mileage, min_m, max_m = normalize(mileage)
    norm_price, min_p, max_p = normalize(price)

    theta0, theta1 = train_model(norm_mileage, norm_price)

    try:
        real_theta0, real_theta1 = denormalize_theta(
            theta0, theta1,
            min_m, max_m, min_p, max_p
        )
    except ValueError as error:
        print(f"Cannot train model: {error}")
        return

    save_thetas(real_theta0, real_theta1, THETAS_FILE)

    print("Training complete!")
    print(f"theta0 = {real_theta0:.6f}")
    print(f"theta1 = {real_theta1:.6f}")


if __name__ == "__main__":
    main()
