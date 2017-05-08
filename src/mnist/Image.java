package mnist;

public class Image {
  public final Size size;
  public final byte[] data;

  public Image(Size size, byte[] data) {
    if (size.count() != data.length) {
      throw new IllegalArgumentException("Image with size " + size
          + " should have data length = " + size.count());
    }
    this.size = size;
    this.data = data;
  }
}
