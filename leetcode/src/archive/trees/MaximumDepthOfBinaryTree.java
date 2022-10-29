package archive.trees;

public class MaximumDepthOfBinaryTree {
    public static void main(String[] args) {
        System.out.println(new MaximumDepthOfBinaryTree().maxDepth(TreeNodeFactory.newBinaryTree(1,2,3,4,5,null,6,7)));
    }

    public int maxDepth(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int left = maxDepth(root.left) + 1;
        int right = maxDepth(root.right) + 1;

        return Math.max(left, right);
    }
}
