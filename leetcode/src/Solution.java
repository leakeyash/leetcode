import helper.TreeNode;

import java.util.Deque;
import java.util.LinkedList;

public class Solution {
    public static void main(String[] args) {
        System.out.println("test");
        System.out.println(
                new Solution()
                        .searchRange(new int[]{1}, 1)
        );
    }
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        while(left <= right) {
            int mid = (left + right) >>> 1;
            if(nums[mid] == target) {
                return mid;
            }
            if(nums[mid] > target) {
                if(nums[left] > target) {
                    return mid;
                }
            }
        }
        return -1;
    }
    public int[] searchRange(int[] nums, int target) {
        int left = binarySearch(nums, target);
        int right = binarySearch(nums, target+1)-1;
        if(left<=right && left>=0 && right <= nums.length-1) {
            return new int[]{left,right};
        }
        return new int[]{-1,-1};
    }

    public int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        while(left <= right) {
            int mid = (left + right + 1) >>> 1;
            if(nums[mid] >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
    public int maxRepeating(String sequence, String word) {
        int left = 0;
        int res = 0;
        int cur = 0;
        while(left<=sequence.length()-word.length()) {
            boolean flag = true;
            for (int i = left; i < left + word.length(); i++) {
                if(sequence.charAt(i) != word.charAt(i - left)) {
                    flag = false;
                    break;
                }
            }
            if(flag){
                left = left + word.length();
                cur++;
            } else {
                res = Math.max(cur, res);
                left = left - cur * word.length() + 1;
                cur = 0;
            }
        }
        return Math.max(cur, res);
    }
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int index1 = 0;
        int index2 = 0;
        while (index1 < word1.length() && index2 < word2.length()) {
            sb.append(word1.charAt(index1));
            sb.append(word2.charAt(index2));
            index1++;
            index2++;
        }
        while (index1 < word1.length()) {
            sb.append(word1.charAt(index1));
            index1++;
        }
        while (index2 < word2.length()) {
            sb.append(word2.charAt(index2));
            index2++;
        }
        return sb.toString();
    }

    public int minCameraCover(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        int level = 1;
        int cnt = 0;
        while (!deque.isEmpty()) {
            boolean flag = level % 2 == 0;
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();

            }
            level++;
        }
        return 1;
    }
}
