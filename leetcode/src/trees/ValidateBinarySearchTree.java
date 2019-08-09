package trees;

import java.util.Stack;

public class ValidateBinarySearchTree {
    public static void main(String[] args) {
        System.out.println(Double.MIN_VALUE);
        System.out.println(new ValidateBinarySearchTree()
                .isValidBST(TreeNodeFactory.newBinaryTree(5,1,4,null,null,3,6)));
        System.out.println(new ValidateBinarySearchTree()
                .isValidBST(TreeNodeFactory.newBinaryTree(-2147483648)));
        System.out.println(new ValidateBinarySearchTree()
                .isValidBST(TreeNodeFactory.newBinaryTree(-2147483648,-2147483648)));
    }
    double last = -Double.MAX_VALUE;
    public boolean isValidBST(TreeNode root) {
        if(root == null){
            return true;
        }
        // 中序遍历
        if(!isValidBST(root.left)){
            return false;
        }
        if(root.val <= last){
            return false;
        }
        last = root.val;
        if(!isValidBST(root.right)){
            return false;
        }
        return true;
    }

    boolean isValidBST1(TreeNode root){
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, pre =null;
        while(p != null || !stack.isEmpty()){
            while(p != null){
                stack.push(p);
                p = p.left;
            }
            TreeNode t = stack.pop();
            if(pre != null && t.val <= pre.val) {
                return false;
            }
            pre = t;
            p = t.right;
        }
        return true;
    }
}

