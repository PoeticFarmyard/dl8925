# week-4
# exp-4 sigmoid function

import numpy as np
def sig(x):
  return 1 / (1 + np.exp(-x))

x = 1.0
print('Applying Sigmoid Activation(%.1f) gives %.1f' % (x, sig(x)))

x = -10.0
print('Applying Sigmoid Activation(%.1f) gives %.1f' % (x, sig(x)))

x = 0.0
print('Applying Sigmoid Activation(%.1f) gives %.1f' % (x, sig(x)))

x = 15
print('Applying Sigmoid Activation(%.1f) gives %.1f' % (x, sig(x)))

x = 2.0
print('Applying Sigmoid Activation(%.1f) gives %.1f' % (x, sig(x)))
