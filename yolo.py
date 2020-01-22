from sacred import Experiment
import test

ex = Experiment(name="yolo", ingredients=(test.test_ingredient,))

@ex.automain
def main():
    test.test_method()
