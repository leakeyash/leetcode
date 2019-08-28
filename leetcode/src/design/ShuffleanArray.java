package design;

import java.util.Arrays;
import java.util.Random;

public class ShuffleanArray {
    class Solution {

        int[] raw;
        int[] nums;
        Random random;

        public Solution(int[] nums) {
            raw = Arrays.copyOf(nums, nums.length);
            random = new Random();
        }

        /** Resets the array to its original configuration and return it. */
        public int[] reset() {
            return raw;
        }

        /** Returns a random shuffling of the array. */
        public int[] shuffle() {
            nums = Arrays.copyOf(raw, raw.length);
            for(int i = nums.length - 1; i > 0; i--){
                int j = random.nextInt(i+1);

                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
            }

            return nums;
        }
    }
}
