import helper.ListNode;
import helper.Node;
import helper.TreeNode;

import java.util.*;
import java.util.logging.Level;

public class Contest {
    public static void main(String[] args) {
        System.out.println("Hello");
        new Contest().destroyTargets(new int[]{1,3,5,2,4,6},2);
    }

    // TLE Weekly Contest 90
    public int[] secondGreaterElement(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        res[0] = getRight(0, nums);
        for (int i = 1; i < len; i++) {
            if(nums[i] == nums[i-1]) {
                res[i] = res[i-1];
            } else {
                res[i] = getRight(i,nums);
            }
        }
        return res;
    }

    private int getRight(int cur, int[] nums) {
        int right = cur+1;
        int cnt =0;
        while (right<nums.length) {
            if(nums[right] > nums[cur]) {
                cnt++;
                if(cnt == 2) {
                    return nums[right];
                }
            }
            right++;
        }
        return -1;
    }
    public int destroyTargets(int[] nums, int space) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int mod = nums[i] % space;
            map.put(mod, map.getOrDefault(mod,0)+1);
        }
        int maxCount = 0;
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if(entry.getValue() > maxCount) {
                maxCount = entry.getValue();
            }
        }
        List<Integer> integers = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if(entry.getValue() == maxCount) {
                integers.add(entry.getKey());
            }
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            int mod = nums[i] % space;
            if(integers.contains(mod)) {
                return nums[i];
            }
        }

        return 0;
    }
    public List<String> twoEditWords(String[] queries, String[] dictionary) {
        int wordLen = queries[0].length();
        if(wordLen <= 2) {
            return Arrays.stream(queries).toList();
        }
        List<String> res = new ArrayList<>();
        for (int i = 0; i < queries.length; i++) {
            for (int j = 0; j < dictionary.length; j++) {
                if(twoSame(queries[i], dictionary[j], wordLen)) {
                    res.add(queries[i]);
                    break;
                }
            }
        }
        return res;
    }

    private boolean twoSame(String a, String b, int len) {
        int diffCount = 0;
        for (int i = 0; i < len; i++) {
            if(a.charAt(i) != b.charAt(i)) {
                diffCount++;
            }
        }
        if(diffCount > 2) {
            return false;
        }
        return true;
    }
    public String oddString(String[] words) {
        int[] cache = new int[words[0].length()-1];
        int[] cache1 = new int[words[0].length()-1];
        int[] cache2 = new int[words[0].length()-1];
        for (int i = 1; i < words[0].length(); i++) {
            cache[i-1] =  words[0].charAt(i) - words[0].charAt(i-1);
        }
        for (int i = 1; i < words[1].length(); i++) {
            cache1[i-1] =  words[1].charAt(i) - words[1].charAt(i-1);
        }
        for (int i = 1; i < words[2].length(); i++) {
            cache2[i-1] =  words[2].charAt(i) - words[2].charAt(i-1);
        }
        if(!Arrays.equals(cache, cache1)) {
            if(Arrays.equals(cache1, cache2)) {
                return words[0];
            }
            if(Arrays.equals(cache, cache2)) {
                return words[1];
            }
        }
        if(!Arrays.equals(cache1, cache2)) {
            return words[2];
        }
        for (int i = 3; i < words.length; i++) {
            for (int j = 1; j < words[i].length(); j++) {
                int diff = words[i].charAt(j) - words[i].charAt(j-1);
                if(diff!=cache[j-1]) {
                    return words[i];
                }
            }
        }
        return null;
    }

}
