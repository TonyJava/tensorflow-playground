package tensorflow;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;

public abstract class GraphTask implements AutoCloseable {
  private final Graph graph;
  private Session session;
  private final List<AutoCloseable> closeables;

  public GraphTask() {
    this.graph = new Graph();
    this.closeables = new ArrayList<>();
    this.closeables.add(graph);
  }

  public static void execute(GraphTask task) {
    task.run();
    task.close();
  }

  public abstract void run();

  public Session session() {
    if (session == null) {
      session = new Session(graph);
    }
    return session;
  }

  public Shape shape(long firstDimensionSize, long... otherDimensionSizes) {
    return Shape.make(firstDimensionSize, otherDimensionSizes);
  }

  public Tensor tensor(Object object) {
    Tensor tensor = Tensor.create(object);
    closeables.add(tensor);
    return tensor;
  }

  public Output add(Output value1, Output value2) {
    return binaryOp("Add", value1, value2);
  }

  public Output sub(Output value1, Output value2) {
    return binaryOp("Sub", value1, value2);
  }

  public Output mul(Output value1, Output value2) {
    return binaryOp("Mul", value1, value2);
  }

  public Output div(Output value1, Output value2) {
    return binaryOp("Div", value1, value2);
  }

  public Output square(Output value) {
    return unaryOp("Square", value);
  }

  public Output matMul(Output value1, Output value2) {
    return binaryOp("MatMul", value1, value2);
  }

  public Output softmax(Output value) {
    return unaryOp("Softmax", value);
  }

  public Output resizeBilinear(Output images, Output size) {
    return binaryOp("ResizeBilinear", images, size);
  }

  public Output expandDims(Output input, Output dim) {
    return binaryOp("ExpandDims", input, dim);
  }

  public Output cast(Output value, DataType dtype) {
    return graph
        .opBuilder("Cast", "Cast")
        .addInput(value)
        .setAttr("DstT", dtype)
        .build()
        .output(0);
  }

  public Output decodeJpeg(Output contents, long channels) {
    return graph.opBuilder("DecodeJpeg", "DecodeJpeg")
        .addInput(contents)
        .setAttr("channels", channels)
        .build()
        .output(0);
  }

  public Output constant(String name, Object value) {
    try (Tensor tensor = Tensor.create(value)) {
      return constant(name, tensor);
    }
  }

  private Output constant(String name, Tensor tensor) {
    return graph
        .opBuilder("Const", name)
        .setAttr("dtype", tensor.dataType())
        .setAttr("value", tensor)
        .build()
        .output(0);
  }

  public Output placeholder(String name, DataType type) {
    return graph
        .opBuilder("Placeholder", name)
        .setAttr("dtype", type)
        .build()
        .output(0);
  }

  public Output variable(String name, DataType type, Shape shape) {
    return graph
        .opBuilder("Variable", name)
        .setAttr("dtype", type)
        .setAttr("shape", shape)
        .build()
        .output(0);
  }

  public Output assignZeros(Output variable) {
    Object zeros = Array.newInstance(Float.TYPE, shapeSizes(variable.shape()));
    Output value = constant("initialize/" + variable.op().name(), zeros);
    return assign(variable, value);
  }

  private int[] shapeSizes(Shape shape) {
    int[] result = new int[shape.numDimensions()];
    for (int i = 0; i < result.length; i++) {
      result[i] = (int) shape.size(i);
    }
    return result;
  }

  public Output assign(Output variable, Output value) {
    return graph
        .opBuilder("Assign", "assign/" + variable.op().name())
        .addInput(variable)
        .addInput(value)
        .build()
        .output(0);
  }

  private Output binaryOp(String type, Output input1, Output input2) {
    return graph
        .opBuilder(type, type)
        .addInput(input1)
        .addInput(input2)
        .build()
        .output(0);
  }

  private Output unaryOp(String type, Output input) {
    return graph
        .opBuilder(type, type)
        .addInput(input)
        .build()
        .output(0);
  }

  public void close() {
    for (int i = closeables.size() - 1; 0 <= i; i--) {
      try {
        closeables.get(i).close();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
