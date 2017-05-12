package tensorflow;

import java.util.Arrays;

import org.tensorflow.Tensor;

public class Debug {
  public static void printTensor(Tensor tensor) {
    int rank = tensor.shape().length;
    switch (rank) {
      case 0:
        print0dTensor(tensor);
        break;
      case 1:
        print1dTensor(tensor);
        break;
      case 2:
        print2dTensor(tensor);
        break;
      default:
        throw new IllegalArgumentException("Unsupported Tensor rank = " + rank);
    }
  }

  private static void print0dTensor(Tensor tensor) {
    System.out.println(tensor.floatValue());
  }

  private static void print1dTensor(Tensor tensor) {
    long[] shape = tensor.shape();
    int rowCount = (int) shape[0];
    float[] result = new float[rowCount];
    tensor.copyTo(result);
    System.out.println(Arrays.toString(result));
  }

  private static void print2dTensor(Tensor tensor) {
    long[] shape = tensor.shape();
    int rowCount = (int) shape[0];
    int columnCount = (int) shape[1];
    float[][] result = new float[rowCount][columnCount];
    tensor.copyTo(result);
    for (int i = 0; i < rowCount; i++) {
      System.out.println(Arrays.toString(result[i]));
    }
  }
}
