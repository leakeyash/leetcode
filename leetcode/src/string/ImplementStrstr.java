package string;

public class ImplementStrstr {
    public static void main(String[] args) {
        int value = new ImplementStrstr().strStr("hello","ll");
        System.out.println(value);
        assert value == 2;
        System.out.println(new ImplementStrstr().strStr("aaaaa","bba"));
        System.out.println(new ImplementStrstr().strStr("aaa","aaaaa"));
        System.out.println(new ImplementStrstr().strStr("a","a"));
        System.out.println(new ImplementStrstr().strStr("mississippi","issip"));

        System.out.println(new ImplementStrstr().strStr1("hello","ll"));
        System.out.println(new ImplementStrstr().strStr1("aaaaa","bba"));
        System.out.println(new ImplementStrstr().strStr1("aaa","aaaaa"));
        System.out.println(new ImplementStrstr().strStr1("a","a"));
        System.out.println(new ImplementStrstr().strStr1("mississippi","issip"));
    }
    public int strStr(String haystack, String needle) {
        if(needle == null || needle.length() == 0){
            return 0;
        }
        char[] needles = needle.toCharArray();
        int srcLength = haystack.length();
        int targetLength = needles.length;
        for(int i=0;i<srcLength;i++){
            if(haystack.charAt(i) == needles[0]){
                int matchCount = 1;
                for(int j=1;j<targetLength;j++){
                    if(i+j <srcLength && haystack.charAt(i+j)==needles[j]){
                        matchCount ++;
                    } else{
                        break;
                    }
                }
                if(matchCount == targetLength){
                    return i;
                }
            }
        }
        return -1;
    }
    public int strStr1(String haystack, String needle){
        char[] source = haystack.toCharArray();
        char[] target = needle.toCharArray();
        int sourceCount = source.length;
        int targetCount = target.length;
        if (0 >= sourceCount) {
            return (targetCount == 0 ? sourceCount : -1);
        }

        if (targetCount == 0) {
            return 0;
        }

        char first = target[0];
        int max = sourceCount - targetCount;

        for (int i = 0; i <= max; i++) {
            /* Look for first character. */
            if (source[i] != first) {
                while (++i <= max && source[i] != first);
            }

            /* Found first character, now look at the rest of v2 */
            if (i <= max) {
                int j = i + 1;
                int end = j + targetCount - 1;
                for (int k = 1; j < end && source[j]
                        == target[k]; j++, k++);

                if (j == end) {
                    /* Found whole string. */
                    return i;
                }
            }
        }
        return -1;
    }

}
