package array;

import java.util.Arrays;

public class MoveZeroes {
    public static void main(String[] args) {
        int[] nums = new int[]{0,1,0,3,12};
        new MoveZeroes().moveZeroes(nums);
        System.out.println(Arrays.toString(nums));
    }

    public void moveZeroes(int[] nums) {
        int zeroCount = 0;
        int index = 0;
        while(index < nums.length){
            if(nums[index]==0){
                zeroCount++;
            } else{
                nums[index-zeroCount] = nums[index];
            }
            index++;
        }
        for(int i=nums.length-zeroCount;i<nums.length;i++){
            nums[i]=0;
        }
    }
}
