from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import json
from flask_sqlalchemy import SQLAlchemy
import math
import numpy as np
import scipy.io as sio

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost:3306/tripshare'
db = SQLAlchemy(app)

CORS(app, resources={r'/api/*': {'origins': '*'}})

@app.route('/num_identifier', methods=['POST'])
@cross_origin()
def num_identifier():
  if request.method == 'POST':
    data = request.get_json()
    image_data = [1] + json.loads(data['image'])
    y = data['value']

    # add_training_data(image_data, y)

    result = determine_number(image_data)
    i = 0
    maxNum = result[result.index(max(result))]
    while i < 10:
      print(i, (result[i]/maxNum) * float(100))
      i += 1

    print('RESULT', result.index(max(result)))

    return jsonify({'prediction': result.index(max(result))})

def add_training_data(image_data, value):
  db.engine.execute(
    'INSERT INTO Number_Training_Data (image, value) \
     VALUES ("%(image_data)s", %(value)d)'
    % {'image_data': image_data, 'value': value}
  )

#  Returns formatted training data and associated Y values
def get_training_data():
  training_data_json = db.engine.execute('SELECT image, value FROM Number_Training_Data')
  training_data_json = training_data_json.fetchall()
  training_data = []
  y = []
  for x in range(0, len(training_data_json)):
    img_data = modify_training_data(json.loads(training_data_json[x]['image']))
    training_data.append([1] + json.loads(training_data_json[x]['image']))
    y.append([training_data_json[x]['value']])

  return {
    'training_data': np.mat(training_data),
    'y': np.mat(y)
  }

# Add new features to existing pixel data
def modify_training_data(array):
  return [1] + new_features(array) # + array

def expand_stroke(array):
  for pixel in array:
    row = pixel / 20
    col = pixel % 20

    if (array[pixel] is 1):
      if (row is not 0):
        array[pixel - 1] = 2 if array[pixel - 1] is 0 else array[pixel - 1]
      if (row is not 19):
        array[pixel + 1] = 2 if array[pixel + 1] is 0 else array[pixel + 1]
      if (col is not 0):
        array[pixel - 20] = 2 if array[pixel - 20] is 0 else array[pixel - 20]
      if (col is not 19):
        array[pixel + 20] = 2 if array[pixel + 20] is 0 else array[pixel + 20]

  for pixel in array:
    array[pixel] = 1 if pixel > 0 else 0

  return array



# Generates various calculated features including in an array:
#   Total pixels: # pixels colored in the image
#   Row count: # rows that have at least one colored pixel
#   Column count: # columns that have at least one colored pixel
#   Row sums (one for each row): # of colored pixels in each row
#   Column sums (one for each column): # of colored pixels in each column
#   Top distribution: % pixels in top half of number pixel boundaries
#   Bottom distribution: % pixels in bottom half of number pixel boundaries
#   Left distribution: % pixels in left half of number pixel boundaries
#   Right distribution: % pixels in right half of number pixel boundaries
#   Top left  distribution: % pixels in top left quarter of number pixel boundaries
#   Top right distribution: % pixels in top right quarter of number pixel boundaries
#   Bottom left distribution: % pixels in bottom left quarter of number pixel boundaries
#   Bottom right distribution: % pixels in bottom right quarter of number pixel boundaries

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

  total_pixels = total_pixels if total_pixels > 0 else 1

  for i in range (0, 19):
    row_count += 1 if row_sums[i] > 0 else 0
    col_count += 1 if col_sums[i] > 0 else 0

  left_bound = None
  right_bound = None
  top_bound = None
  bottom_bound = None

  # Get boundaries of number image based on colored pixels
  for i in range(0, 20):
    if row_sums[i] > 0:
      if left_bound is None:
        left_bound = i
      right_bound = i

    if col_sums[i] > 0:
      if top_bound is None:
        top_bound = i
      bottom_bound = i

  left_bound = left_bound if left_bound else 0
  right_bound = right_bound if right_bound else 0
  top_bound = top_bound if top_bound else 0
  bottom_bound = bottom_bound if bottom_bound else 0
  horizontal_mid_index = (left_bound + right_bound) / 2
  vertical_mid_index = (top_bound + bottom_bound) / 2

  top_left_count = 0
  top_right_count = 0
  bottom_left_count = 0
  bottom_right_count = 0

  left_count = 0
  right_count = 0
  top_count = 0
  bottom_count = 0

  # Determine pixel distribution based on number image dimesions
  for pixel in range(0, len(img_array)):
    row = pixel / 20
    col = pixel % 20

    if row < horizontal_mid_index and col < vertical_mid_index:
      top_left_count += img_array[pixel]
    elif row < horizontal_mid_index and col > vertical_mid_index:
      top_right_count += img_array[pixel]
    elif row > horizontal_mid_index and col < vertical_mid_index:
      bottom_left_count += img_array[pixel]
    elif row > horizontal_mid_index and col > vertical_mid_index:
      bottom_right_count += img_array[pixel]

    if row < horizontal_mid_index:
      top_count += img_array[pixel]
    elif row > horizontal_mid_index:
      bottom_count += img_array[pixel]
    if col < vertical_mid_index:
      left_count += img_array[pixel]
    elif col > vertical_mid_index:
      right_count += img_array[pixel]

  pixel_distribution = [
    top_count / float(total_pixels),
    bottom_count / float(total_pixels),
    left_count / float(total_pixels),
    right_count / float(total_pixels),
    top_left_count / float(total_pixels),
    top_right_count / float(total_pixels),
    bottom_left_count / float(total_pixels),
    bottom_right_count / float(total_pixels),
    top_count / float(bottom_count if bottom_count else 1),
    left_count / float(right_count if right_count else 1)
  ]

  return [total_pixels, row_count, col_count] + row_sums + col_sums + pixel_distribution

# Return theta values for each Y value
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
  for iter in range(0, num_iters):
    # Run gradient descent to optimize thetas for regression
    h = sigmoid(X * theta)
    theta = theta - sum(np.multiply((h-y), X), 0).T * (alpha / m)
    print(num, iter, logistic_cost(X, y, theta))

  for feature in range(0, theta.shape[0]):
    db.engine.execute(
      'REPLACE INTO Number_Thetas (feature, coefficient, value) \
      VALUES (%(feature)d, %(coefficient)f, %(value)d)' %
      {'feature': feature, 'coefficient': theta[feature], 'value': num})

# Sigmoid returns a value between 0 and 1
# When z = 0, should return 0.5
# def sigmoid(z):
#   return 1 / (1 + math.exp(-1 * z))

# Returns inputs and outputs for training and cross validation data
def get_training_and_cv_data(use_3rd_party_training_data = 0):
  data = get_training_data()

  # Use 3rd party data
  if use_3rd_party_training_data:
    # If we use third party data for training, we use all of our personal data for cross validation
    personal_data_training_size = 0
    convert_grayscale_to_binary = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
    # 3rd party data uses Y label of 10 for zero since it plays nicely in Octave
    # This function simply changes the label to 0 to make it more consistent
    replace_10_with_0 = np.vectorize(lambda y: 0 if y == 10 else y)
    data_3rd_party = sio.loadmat('data.mat')
    training_X = convert_grayscale_to_binary(data_3rd_party['X'])
    training_y = replace_10_with_0(data_3rd_party['y'])
  # Use personal data
  else:
    personal_data_training_size = len(data['training_data']) * 0.8
    training_X = data['training_data'][0: personal_data_training_size]
    training_y = data['y'][0: personal_data_training_size]

  cross_validation_X = data['training_data'][personal_data_training_size: len(data['training_data'])]
  cross_validation_y = data['y'][personal_data_training_size: len(data['training_data'])]

  return {
    'training_X': training_X,
    'training_y': training_y,
    'cross_validation_X': cross_validation_X,
    'cross_validation_y': cross_validation_y
  }


@app.route('/run_regression', methods=['GET'])
@cross_origin()
def run_regression():
  data = get_training_and_cv_data(0)

  theta = [[0]] * data['training_X'].shape[1]
  theta = np.mat(theta)
  y_convert_function = np.vectorize(lambda y, num: 1 if y == num else 0)

  for num in range(0, 10):
    gradient_descent(num, data['training_X'], y_convert_function(data['training_y'], num), theta, 0.03, 400)
  check_results(data['cross_validation_X'], data['cross_validation_y'])
  return 'success'


sigmoid = np.vectorize(lambda z: 1 / (1 + math.exp(-1 * z)))

def check_results(cv_X, cv_y):
  # data = get_training_data()
  theta = get_thetas()

  output = []
  for cv_data_row in range(0, cv_X.shape[0]):
    X_data = cv_X[cv_data_row]
    cur_results =  []
    for i in range(0, len(theta)):
      cur_results.append(sigmoid(X_data * theta[i]).A1[0])

    output.append([cv_y[cv_data_row].A1[0], cur_results])

  correct = 0
  total = 0
  for i in range(0, len(output)):
    total += 1
    if (output[i][0] == output[i][1].index(max(output[i][1]))):
      correct += 1

  print('correct', correct)
  print('total', total)
  print('accurracy %f' % (correct/float(total)))


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
