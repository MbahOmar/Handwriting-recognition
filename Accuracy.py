import numpy as np
import matplotlib.pyplot as plt
#import  RunningTheModel
# Create model.
from RunningTheModel import neural_net,x_test,y_test,accuracy
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
n_images = 200
test_images = x_test[:n_images]
test_labels = y_test[:n_images]
predictions = neural_net(test_images)

for i in range(n_images):
    model_prediction = np.argmax(predictions.numpy()[i])
    if (model_prediction != test_labels[i]):
        plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray_r')
        plt.show()
        print("Original Labels: %i" % test_labels[i])
        print("Model prediction: %i" % model_prediction)