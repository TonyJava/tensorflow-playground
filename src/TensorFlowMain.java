import static tensorflow.GraphTask.execute;

import java.util.Arrays;

import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import tensorflow.GraphTask;

// https://www.tensorflow.org/install/install_java
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
public class TensorFlowMain {
  public static void main(String[] args) throws Exception {
    execute(new GraphTask() {
      public void run() {
        Output c1 = constant("Const1", new int[] { 1, 2, 3 });
        Output c2 = constant("Const2", new int[] { 2, 4, 8 });
        Output sum = add(c1, c2);
        Output square = square(sum);
        Session session = session();
        Tensor output = session.runner().fetch(square).run().get(0);
        System.out.println(output.dataType());
        System.out.println(Arrays.toString(outputToIntArray(output)));
      }
    });
  }

  private static int[] outputToIntArray(Tensor output) {
    int[] result = new int[3];
    output.copyTo(result);
    return result;
  }
}
