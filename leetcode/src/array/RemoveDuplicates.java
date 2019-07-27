package array;

public class RemoveDuplicates {
    public static void main(String[] args) {
        int[] nums = new int[]{0,0,1,1,1,2,2,3,3,4};
        System.out.println(new RemoveDuplicates().removeDuplicates(nums));
        printIntArray(nums);
    }

    public int removeDuplicates(int[] nums) {
        int count =0;
        for(int i=1; i< nums.length;i++){
            if(nums[i-1] != nums[i]){
                count ++;
            }
            nums[count] = nums[i];
        }
        return count+1;
    }

    public static void printIntArray(int[] array){
        for(int item: array){
            System.out.println(item);
        }
    }
}
