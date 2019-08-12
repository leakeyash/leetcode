package sortingandsearching;

public class MergeSortedArray {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        m = m - 1;
        n = n - 1;
        for(int i = m + n + 1;i>=0;i--){
            int val1 = m >= 0 ? nums1[m]: Integer.MIN_VALUE;
            int val2 = n >= 0 ? nums2[n]:Integer.MIN_VALUE;
            if(val1 == Integer.MIN_VALUE && val2 == Integer.MIN_VALUE){
                break;
            }
            if(val1 > val2){
                nums1[i] = val1;
                m --;
            } else{
                nums1[i] = val2;
                n--;
            }
        }
    }

    public void merge1(int[] nums1, int m, int[] nums2, int n) {
        if (nums2 == null) {
            return;
        }

        while (m > 0 && n > 0) {
            if (nums1[m-1] > nums2[n-1]) {
                nums1[m+n-1] = nums1[m-1];
                m--;
            } else {
                nums1[m+n-1] = nums2[n-1];
                n--;
            }
        }
        if (m == 0) {
            if (n > 0) {
                System.arraycopy(nums2, 0, nums1, 0, n);
            }
        }
    }
}
