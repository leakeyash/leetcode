package archive.array;

public class Rotate {
    public static void main(String[] args) {
        int[] nums = new int[]{1,2,3,4,5,6,7};
        new Rotate().rotate(nums,3);
        for(int item: nums){
            System.out.print(item);
            System.out.print(',');
        }
    }

    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        if(k == 0){
            return;
        }

        int index = 0;
        int count = 0;
        int dest;
        int temp = nums[index];
        int start = 0;
        while(count < nums.length){
            index = (index + k)%nums.length;
            dest = nums[index];
            nums[index] = temp;
            temp = dest;
            if(index == start){
                start++;
                temp = nums[start];
                index = start;
            }
            count++;
        }
    }
}
