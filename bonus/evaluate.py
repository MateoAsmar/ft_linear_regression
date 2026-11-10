import csv
import math
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "mandatory" / "data.csv"
THETAS_FILE = ROOT_DIR / "mandatory" / "thetas.csv"


def read_data(filename):
    mileage, price = [], []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        try:
            next(reader)
        except StopIteration:
            raise ValueError("Dataset is empty.") from None
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


def load_thetas(filename):
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        try:
            next(reader)
            row = next(reader)
            theta0, theta1 = float(row[0]), float(row[1])
        except (StopIteration, IndexError, ValueError, csv.Error):
            raise ValueError("Theta file must contain numeric theta0 and theta1 values.") from None
    if not math.isfinite(theta0) or not math.isfinite(theta1):
        raise ValueError("Theta values must be finite numbers.")
    return theta0, theta1


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def compute_mse(mileage, price, theta0, theta1):
    if len(mileage) != len(price):
        raise ValueError("Mileage and price must contain the same number of values.")
    m = len(price)

    if m == 0:
        raise ValueError("Dataset is empty.")

    total_error = 0.0

    for i in range(m):
        prediction = estimate_price(mileage[i], theta0, theta1)
        total_error += (prediction - price[i]) ** 2

    return total_error / m


def main():
    try:
        mileage, price = read_data(DATA_FILE)
        theta0, theta1 = load_thetas(THETAS_FILE)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Cannot evaluate model: {error}")
        return

    mse = compute_mse(mileage, price, theta0, theta1)

    print(f"Mean Squared Error: {mse:.2f}")


if __name__ == "__main__":
    main()
