package archive.array;

import java.util.Arrays;
import java.util.HashMap;

public class TwoSum {
    public static void main(String[] args) {
        int[] numbers = new int[]{2, 7, 11, 15};
        int target = 9;
        System.out.println(Arrays.toString(new TwoSum().twoSum(numbers, target)));
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> temp = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (temp.containsKey(target - nums[i])) {
                return new int[]{temp.get(target - nums[i]), i};
            } else {
                temp.put(nums[i], i);
            }
        }
        return new int[]{};
    }
}
