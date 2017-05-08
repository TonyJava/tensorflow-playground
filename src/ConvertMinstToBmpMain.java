import java.io.IOException;
import java.util.List;

import mnist.MnistReader;
import mnist.MnistNumber;
import mnist.MnistWriter;

public class ConvertMinstToBmpMain {
  private static final String OUTPUT_DIR = "/home/mikoch/tmp/mnist/";

  public static void main(String... args) throws IOException {
    List<MnistNumber> trainingNumbers = MnistReader.readTrainingSet();
    MnistWriter.saveMnist(trainingNumbers, OUTPUT_DIR);
  }
}
