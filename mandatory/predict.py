import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
THETAS_FILE = BASE_DIR / "thetas.csv"


def load_thetas(filename):
    if not filename.exists():
        raise FileNotFoundError(filename)

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


def main():
    try:
        theta0, theta1 = load_thetas(THETAS_FILE)
    except FileNotFoundError:
        print("Model not found. Run 'python3 mandatory/train.py' first.")
        return
    except (OSError, ValueError) as error:
        print(f"Cannot load model: {error}")
        return

    try:
        mileage = float(input("Enter car mileage (in km): "))

        if not math.isfinite(mileage):
            print("Mileage must be a finite number.")
            return
        if mileage < 0:
            print("Mileage cannot be negative.")
            return

    except ValueError:
        print("Invalid input. Please enter a numeric mileage.")
        return

    price = estimate_price(mileage, theta0, theta1)

    print(f"Estimated car price: ${price:.2f}")


if __name__ == "__main__":
    main()
