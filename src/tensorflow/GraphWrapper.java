package tensorflow;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;

public class GraphWrapper implements AutoCloseable {
  private final Graph graph;

  public GraphWrapper() {
    this(new Graph());
  }

  public GraphWrapper(Graph graph) {
    this.graph = graph;
  }

  public Session newSession() {
    return new Session(graph);
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
      return graph
          .opBuilder("Const", name)
          .setAttr("dtype", tensor.dataType())
          .setAttr("value", tensor)
          .build()
          .output(0);
    }
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

  public Output assign(Output variable, Output value) {
    return graph
        .opBuilder("Assign", "assign->" + variable.op().name())
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
    graph.close();
  }
}
