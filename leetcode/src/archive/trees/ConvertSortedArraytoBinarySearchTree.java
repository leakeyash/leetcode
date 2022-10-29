package archive.trees;

public class ConvertSortedArraytoBinarySearchTree {
    public TreeNode sortedArrayToBST(int[] nums) {
        return construct(nums,0,nums.length-1);
    }

    private TreeNode construct(int[] arr,int low,int high) {
        if(low>high) {
            return null;
        }
        int mid=(low+high)/2;
        TreeNode node=new TreeNode(arr[mid]);

        node.left=construct(arr, low, mid-1);
        node.right=construct(arr, mid+1, high);
        return node;
    }
}
