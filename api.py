from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import json
from flask_sqlalchemy import SQLAlchemy
import math
import numpy as np
import scipy.io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost:3306/tripshare'
db = SQLAlchemy(app)


CORS(app, resources={r'/api/*': {'origins': '*'}})

@app.route('/num_identifier', methods=['GET', 'POST'])
@cross_origin()
def num_identifier():
  if request.method == 'POST':
    data = request.get_json()
    img_data = [1] + json.loads(data['image'])
    y = data['value']

    db.engine.execute('INSERT INTO Number_Training_Data (image, value) VALUES ("%(img_data)s", %(value)d)' % {'img_data': data['image'], 'value': data['value']})


    result = determine_number(img_data)
    print(result)
    print('RESULT', result.index(max(result)))

    return jsonify(result.index(max(result)))
  else:
    return 'GET REQUEST'

# def one_vs_all_regression():
  # Each image is 100x100 pixels, pluts one
  # features = 100 * 100 + 1
  # 10 labels 0 to 10
  # num_labels = 10

#  Returns formatted training data and associated Y values
def get_training_data():
  training_data_json = db.engine.execute('SELECT image, value FROM Number_Training_Data')
  training_data_json = training_data_json.fetchall()
  training_data = []
  y = []
  for x in range(0, len(training_data_json)):
    img_data = modify_training_data(json.loads(training_data_json[x]['image']))
    # img_data = [1] + json.loads(training_data_json[x]['image'])
    training_data.append([1] + json.loads(training_data_json[x]['image']))
    y.append([training_data_json[x]['value']])

  return {
    'training_data': np.mat(training_data),
    'y': np.mat(y)
  }

def modify_training_data(array):
  return [1] + new_features(array) # + array

def new_features(img_array):
  # represent the number density (sum of 1s) in each row/col
  row_sums = [0] * 20
  col_sums = [0] * 20

  # represent the count of rows/columns with at least one marked pixel
  row_count = 0
  col_count = 0

  total_pixels = 0

  for pixel in range(0, len(img_array)):
    row = pixel / 20
    col = pixel % 20
    row_sums[row] += img_array[pixel]
    col_sums[col] += img_array[pixel]
    total_pixels += img_array[pixel]

  for i in range (0, 19):
    row_count += 1 if row_sums[i] > 0 else 0
    col_count += 1 if col_sums[i] > 0 else 0

  return [total_pixels, row_count, col_count] + row_sums + col_sums


def get_thetas():
  output = []

  for num in range(0, 10):
    thetas = db.engine.execute('SELECT coefficient FROM Number_Thetas WHERE value = %d ORDER BY feature ASC' % num)
    thetas = thetas.fetchall()
    output.append(np.mat(thetas))

  return output

# X: training example inputs
# y: training example outputs
# theta: regression coefficients
def logistic_cost(X, y, theta, lam = 0):
  # m represents the number of training examples
  m = y.shape[0]
  square = np.vectorize(lambda x: x*x)
  sigmoid = np.vectorize(lambda z: 1 / (1 + math.exp(-1 * z)))
  print('A', X.shape)
  print('B', theta.shape)
  h = sigmoid((X * theta))

  cost = np.sum(np.multiply(-y, (np.log(h))) - np.multiply((1 - y), np.log(1 - h))) / m
  regularization_term = lam * np.sum(np.square(theta))

  return cost + regularization_term


# X: training example inputs
# y: training example outputs
# theta: regression coefficients
# alpha: learning rate
# num_iters: number if iterations of GD to run
def gradient_descent(num, X, y, theta, alpha = 0.01, num_iters = 400):
  # m represents the number of training examples
  m = y.shape[0]
  sigmoid = np.vectorize(lambda z: 1 / (1 + math.exp(-1 * z)))
  print('AA', X.shape)
  print('BB', theta.shape)
  for iter in range(0, num_iters):
    # Run gradient descent to optimize thetas for regression
    h = sigmoid(X * theta)
    theta = theta - sum(np.multiply((h-y), X), 0).T * (alpha / m)
    print(num, iter, logistic_cost(X, y, theta))

  for feature in range(0, theta.shape[0]):
    db.engine.execute(
      'REPLACE INTO Number_Thetas (feature, coefficient, value) VALUES (%(feature)d, %(coefficient)f, %(value)d)' % {'feature': feature, 'coefficient': theta[feature], 'value': num})





# Sigmoid returns a value between 0 and 1
# When z = 0, should return 0.5
# def sigmoid(z):
#   return 1 / (1 + math.exp(-1 * z))

@app.route('/run_regression', methods=['GET'])
@cross_origin()
def run_regression():
  data = get_training_data()

  # Split data into training set and cross validation set
  training_data_size = len(data['training_data']) * 0.8
  training_data_X = data['training_data'][0: training_data_size]
  training_data_y = data['y'][0: training_data_size]
  cross_validation_data_X = data['training_data'][training_data_size: len(data['training_data'])]
  cross_validation_y = data['y'][training_data_size: len(data['training_data'])]

  theta = [0] * training_data_X.shape[1]
  theta = np.mat(theta)
  yConvertFunc = np.vectorize(lambda y, num: 1 if y == num else 0)

  for num in range(0, 10):
    gradient_descent(num, data['training_data'], yConvertFunc(data['y'], num), theta, 0.01)

  # check_results(cross_validation_data_X)
  return 'success'


sigmoid = np.vectorize(lambda z: 1 / (1 + math.exp(-1 * z)))

@app.route('/check_results', methods=['GET'])
@cross_origin()
def check_results(cross_validation_set):
  data = get_training_data()
  X = data['training_data']
  y = data['y'].A1
  theta = get_thetas()

  output = []
  for iter in range(0, len(X)):

    X_data = np.mat(X[iter])
    cur_results =  []
    for i in range(0, len(theta)):
      cur_results.append(sigmoid(X_data * theta[i]))
    output.append([y[iter], cur_results])

  print(output)
  correct = 0
  total = 0
  for i in range(0, len(output)):
    total += 1
    if (output[i][0] == output[i][1]):
      correct += 1

  print('correct', correct)
  print('total', total)
  print('accurracy %f' % correct/total)
  return 'GET REQUEST'



def determine_number(X):
  theta = get_thetas()

  X = np.mat(X)
  results = []
  for i in range(0, len(theta)):
    results.append(sigmoid(X * theta[i]).A1[0])

  return results



# check_results()


if __name__ == '__main__':
  app.run(debug=True)
