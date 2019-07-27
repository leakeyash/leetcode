package array;

import java.util.Arrays;

public class PlusOne {
    public static void main(String[] args) {
        int[] digits = new int[]{4,3,2,1};
        System.out.println(Arrays.toString(new PlusOne().plusOne(digits)));
        digits = new int[]{4,3,2,9};
        System.out.println(Arrays.toString(new PlusOne().plusOne(digits)));
        digits = new int[]{9,9,9,9};
        System.out.println(Arrays.toString(new PlusOne().plusOne(digits)));
    }

    public int[] plusOne(int[] digits) {
        int index = digits.length - 1;
        boolean flag = true;
        while (index >= 0) {
            if(flag){
                if(digits[index] == 9){
                    digits[index] = 0;
                } else{
                    digits[index] += 1;
                    flag = false;
                }
            }
            index--;
        }
        if(flag){
            int[] result = new int[digits.length+1];
            System.arraycopy(digits,0,result,1,digits.length);
            result[0]=1;
            return result;
        }else{
            return digits;
        }
    }
}
