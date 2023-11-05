from Network1 import mnist_loader
from Network2 import network2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
    net.save(filename="trainedNet")
