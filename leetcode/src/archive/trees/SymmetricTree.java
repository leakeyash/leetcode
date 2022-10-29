package archive.trees;

import java.util.*;

public class SymmetricTree {
    public static void main(String[] args) {
        System.out.println(new SymmetricTree()
                .isSymmetric(TreeNodeFactory.newBinaryTree(1,2,2,3,4,4,3)));
        System.out.println(new SymmetricTree()
                .isSymmetric(TreeNodeFactory.newBinaryTree(1,2,2,2,null,2)));
    }

    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }
    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return (t1.val == t2.val)
                && isMirror(t1.right, t2.left)
                && isMirror(t1.left, t2.right);
    }

    public boolean isSymmetric1(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode treeNode1 = queue.poll();
            TreeNode treeNode2 = queue.poll();
            if(treeNode1 ==null && treeNode2 == null){
                continue;
            }
            if(treeNode1 == null || treeNode2 == null){
                return false;
            }
            if(treeNode1.val != treeNode2.val){
                return false;
            }
            queue.add(treeNode1.left);
            queue.add(treeNode2.right);
            queue.add(treeNode1.right);
            queue.add(treeNode2.left);
        }
        return true;
    }


}
