import helper.ListNode;
import helper.TreeNode;

import java.util.*;

public class Solution {
    public static void main(String[] args) {
        System.out.println("test");
        System.out.println(
                new Solution()
                        .findMin(new int[]{3,1,2})
        );
    }
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int pre1 = Integer.MAX_VALUE, pre2 = Integer.MAX_VALUE, pre3 = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            int cur = nums[i];
            if(pre1 == cur) {
                continue;
            }
            pre1 = cur;
            int sum1 = -cur;
            pre2 = Integer.MAX_VALUE;
            for (int j = i + 1; j < nums.length; j++) {
                int cur2 = nums[j];
                if(pre2 == cur2) {
                    continue;
                }
                pre2 = cur2;
                int sum2 = sum1 - nums[j];
                for (int k = j+1; k < nums.length; k++) {
                    if(nums[k] == sum2) {
                        List<Integer> l = new ArrayList<>();
                        l.add(cur);
                        l.add(cur2);
                        l.add(nums[k]);
                        res.add(l);
                        break;
                    }
                }
            }
        }
        return res;
    }
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) {
            return null;
        }
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode tmp = res;
        while (tmp.next != null && tmp.next.next != null) {
            if(tmp.next.val == tmp.next.next.val) {
                int val = tmp.next.val;
                while (tmp.next != null && tmp.next.val == val) {
                    tmp.next = tmp.next.next;
                }
            } else {
                tmp = tmp.next;
            }
        }
        return res.next;
    }
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if(nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if(nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int rl = 0, rh = m-1;
        while (rl <= rh) {
            int mid = (rl + rh)>>>1;
            if(matrix[mid][0] == target) {
                return true;
            } else if(matrix[mid][0] > target) {
                rh = mid - 1;
            } else {
                rl = mid + 1;
            }
        }
        int row = rh;
        if(row < 0) {
            return false;
        }
        int cl = 0, cr = n -1;
        while (cl <= cr) {
            int mid = (cl + cr)>>>1;
            int cur = matrix[row][mid];
            if(cur == target) {
                return true;
            } else if (cur > target) {
                cr = mid -1;
            } else {
                cl = mid + 1;
            }
        }
        return false;
    }
    public int mySqrt(int x) {
        if(x == 0 || x == 1) {
            return x;
        }
        int left = 0, right = x;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            long cur = (long) mid * mid;
            if(cur == x) {
                return mid;
            } else if (cur > x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left - 1;
    }
    public int missingNumber(int[] nums) {
        Arrays.sort(nums);
        int left =0, right = nums.length -1;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            if(nums[mid] == mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right + 1;
    }
    public boolean halvesAreAlike(String s) {
        List<Character> characters = List.of('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U');
        int n = s.length();
        int sum = 0;
        for (int i = 0; i < n/2; i++) {
            sum += characters.contains(s.charAt(i)) ? 1 : 0;
            sum -= characters.contains(s.charAt(i+n/2)) ? 1 : 0;
        }
        return sum == 0;
    }
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        while(left <= right) {
            int mid = (left + right) >>> 1;
            if(nums[mid] == target) {
                return mid;
            }
            if(nums[mid] >= nums[0]) {
                if(nums[mid] > target && target >= nums[0]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if(nums[mid] < target && target <= nums[nums.length-1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
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
