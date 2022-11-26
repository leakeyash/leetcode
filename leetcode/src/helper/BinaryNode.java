package helper;

import java.util.ArrayList;
import java.util.List;

public class BinaryNode {
    public BinaryNode parent;
    public List<BinaryNode> nodes = new ArrayList<>();
    public int val;


    public BinaryNode(int val) {
        this.val = val;
    }
}
