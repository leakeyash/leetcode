package archive.array;

import java.util.*;

public class Intersect {
    public static void main(String[] args) {
        int[] nums1= new int[]{1,2,2,1};
        int[] nums2= new int[]{2,2};
        int[] result = new Intersect().intersect(nums1,nums2);
        System.out.println(Arrays.toString(result));
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int index1 = 0;
        int index2 = 0;
        int count = 0;
        while(index1 < nums1.length && index2 < nums2.length){
            int num1 = nums1[index1];
            int num2 = nums2[index2];
            if(num1==num2){
                index1++;
                index2++;
                nums1[count] = num1;
                count++;
            } else if (num1 < num2){
                index1 ++;
            } else {
                index2 ++;
            }
        }
        int[] nums = new int[count];
        System.arraycopy(nums1,0,nums,0,count);
        return nums;
    }
}
