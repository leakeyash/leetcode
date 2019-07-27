package array;

import java.util.HashSet;
import java.util.Set;

public class ContainsDuplicate {
    public static void main(String[] args) {
        int[] nums = new int[]{3,3};
        System.out.println(new ContainsDuplicate().containsDuplicate(nums));
    }
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> temp = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if(temp.contains(nums[i])){
                return true;
            } else{
                temp.add(nums[i]);
            }
        }
        return false;
    }
}
