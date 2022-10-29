package archive.trees;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class BinaryTreeLevelOrderTraversal {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null){
            return result;
        }
        Queue<TreeNode> treeNodes = new LinkedList<>();
        treeNodes.add(root);
        while (!treeNodes.isEmpty()){
            List<Integer> levelNodes = new ArrayList<>();
            int size = treeNodes.size();
            int i = 0;
            while(i < size){
                TreeNode t = treeNodes.poll();
                levelNodes.add(t.val);
                if(t.left!=null){
                    treeNodes.add(t.left);
                }
                if(t.right!=null){
                    treeNodes.add(t.right);
                }
                i++;
            }
            result.add(levelNodes);
        }
        return result;
    }
}
