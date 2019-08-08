package trees;

public class TreeNodeFactory {

    public static TreeNode newBinaryTree(Integer... values){
        TreeNode result = createBinaryTree(values, 0);
        printTreeAsHierarchy(result);
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

    private static void printTreeAsHierarchy(TreeNode... treeNodes){
        if(treeNodes == null){
            return;
        }
        int nullCount = 0;
        int size = treeNodes.length;
        StringBuilder sb = new StringBuilder();
        TreeNode[] temps = new TreeNode[2*size];
        for (int i = 0; i < size; i++) {
            TreeNode treeNode = treeNodes[i];
            if(i!=0){
                sb.append("->");
            }
            if(treeNode == null){
                temps[2*i] = null;
                temps[2*i + 1] = null;
                nullCount ++;
                sb.append("null");
            } else{
                sb.append(treeNode.val);
                temps[2*i] = treeNode.left;
                temps[2*i + 1] = treeNode.right;
            }
        }
        if(nullCount != size){
            System.out.println(sb);
            printTreeAsHierarchy(temps);
        }
    }
}
