package string;

public class ReverseInteger {
    public static void main(String[] args) {
        System.out.println(new ReverseInteger().reverse(1163847412));
    }

    public int reverse(int x) {
        char[] maxInt = String.valueOf(Integer.MAX_VALUE).toCharArray();
        System.out.println(Integer.MAX_VALUE);
        char[] chars = String.valueOf(x).toCharArray();
        int start = 0;
        int numLength = chars.length;
        if (chars[0] == '-') {
            start = 1;
            maxInt[maxInt.length - 1] += 1;
            numLength -= 1;
        }
        boolean sameLength = maxInt.length == numLength;
        for (int i = start; i < (chars.length) / 2 + start; i++) {
            char temp = chars[i];
            int destIndex = chars.length - 1 - i + start;
            chars[i] = chars[destIndex];
            chars[destIndex] = temp;
        }
        if (sameLength) {
            int index = 0;
            while (index < chars.length) {
                if (chars[index + start] == maxInt[index]) {
                    index++;
                } else if (chars[index + start] > maxInt[index]) {
                    return 0;
                } else {
                    break;
                }
            }
        }
        return Integer.valueOf(String.valueOf(chars));
    }

    public int reverse2(int x) {
        int result = 0;
        int tail;
        while (x != 0) {
            tail = x % 10;
            x /= 10;
            if (result > Integer.MAX_VALUE / 10) {
                return 0;
            }
            if (result == Integer.MAX_VALUE / 10 && tail > 7) {
                return 0;
            }
            if (result < Integer.MIN_VALUE / 10) {
                return 0;
            }
            if (result == Integer.MIN_VALUE / 10 && tail < -8) {
                return 0;
            }

            result = result * 10 + tail;
        }
        return result;
    }
}
