# ft_linear_regression

Train a simple linear regression model with gradient descent to estimate a
car's price from its mileage. The training dataset is `mandatory/data.csv`.

## Requirements

- Python 3
- `matplotlib` for the optional visualization: `python3 -m pip install matplotlib`

Training and prediction use only Python's standard library.

## Mandatory Part

### Training

Run the training program:

    python3 mandatory/train.py

This reads `mandatory/data.csv`, trains the model using gradient descent, and
saves `theta0` and `theta1` in `mandatory/thetas.csv`.

### Prediction

Run the prediction program:

    python3 mandatory/predict.py

Run training first, then enter a car's mileage in kilometers to receive its
estimated price. Negative mileage is rejected.

## Bonus Part

### Visualization

    python3 bonus/plot.py

Displays the dataset and the trained regression line. Run training first.

### Model Evaluation

    python3 bonus/evaluate.py

Calculates the Mean Squared Error (MSE) on the training dataset. Run training
first.

## Hypothesis

estimatePrice(mileage) = theta0 + (theta1 * mileage)

## Algorithm

Simple Linear Regression
Gradient Descent
