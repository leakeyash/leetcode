package helper;

public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public  TreeNode() {
    }

    public TreeNode(int val) {
        this.val = val;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    public static TreeNode newBinaryTree(Integer... values){
        TreeNode result = createBinaryTree(values, 0);
        return result;
    }

    private static TreeNode createBinaryTree(Integer[] values, int index){
        if(index<values.length && values[index] != null){
            TreeNode treeNode = new TreeNode(values[index]);
            treeNode.left = createBinaryTree(values, 2*index + 1);
            treeNode.right = createBinaryTree(values, 2*index + 2);
            return treeNode;
        }
        return null;
    }
}
