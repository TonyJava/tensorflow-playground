import static org.tensorflow.DataType.FLOAT;
import static tensorflow.GraphTask.execute;

import java.util.List;

import org.tensorflow.Output;
import org.tensorflow.Tensor;

import mnist.MnistNumber;
import mnist.MnistReader;
import tensorflow.Debug;
import tensorflow.GraphTask;

public class MnistMain {
  private static final int BATCH_SIZE = 100;

  public static void main(String[] args) throws Exception {
    List<MnistNumber> trainingSet = MnistReader.readTestSet();

    execute(
        new GraphTask() {
          public void task() {
            Output x = placeholder("x", FLOAT);
            Output w = variable("w", FLOAT, shape(784, 10));
            Output b = variable("b", FLOAT, shape(10));
            Output mul = matMul(x, w);
            Output y = softmax(add(mul, b));
            Output y_ = placeholder("y_", FLOAT);

            run(assignZeros(w));
            run(assignZeros(b));

            Output crossEntropy = reduceMean(
                negative(
                    reduceSum(
                        mul(y_, log(y)),
                        constantIntArray(1))),
                constantIntArray(0));

            Tensor trainingBatch = tensor(getTrainingBatch(trainingSet, 0, BATCH_SIZE));
            Tensor labelBatch = tensor(getLabelBatch(trainingSet, 0, BATCH_SIZE));

            Tensor output = session().runner()
                .feed(x, trainingBatch)
                .feed(y_, labelBatch)
                .fetch(crossEntropy)
                .run()
                .get(0);

            Debug.printTensor(output);
          }
        });
  }

  private static float[][] getTrainingBatch(List<MnistNumber> trainingSet, int index, int size) {
    // tf.float32, [None, 784]
    float[][] result = new float[size][];
    for (int i = 0; i < size; i++) {
      result[i] = trainingSet.get(index + i).image.asFloatArray();
    }
    return result;
  }

  private static float[][] getLabelBatch(List<MnistNumber> trainingSet, int index, int size) {
    // tf.float32, [None, 10]
    float[][] result = new float[size][];
    for (int i = 0; i < size; i++) {
      float[] label = new float[10];
      label[trainingSet.get(index + i).label] = 1.0f;
      result[i] = label;
    }
    return result;
  }
}
