package string;

public class AToI {
    private int maxInt = 2147483647;

    public static void main(String[] args) {
        System.out.println('2' - '0');
        System.out.println(Integer.MAX_VALUE);
        //42
        System.out.println(new AToI().myAtoi("42"));
        //-42
        System.out.println(new AToI().myAtoi("   -42"));
        //4193
        System.out.println(new AToI().myAtoi("4193 with words"));
        //0
        System.out.println(new AToI().myAtoi("words and 987"));
        //-2147483648
        System.out.println(new AToI().myAtoi("-91283472332"));
        //-2147483647
        System.out.println(new AToI().myAtoi("-2147483647"));
        //12345678
        System.out.println(new AToI().myAtoi("  0000000000012345678"));
        //-12345678
        System.out.println(new AToI().myAtoi("  -0000000000012345678"));
        //0
        System.out.println(new AToI().myAtoi("000000000000000000"));
        //0
        System.out.println(new AToI().myAtoi("+-2"));
    }

    public int myAtoi(String str) {
        char[] chars = str.toCharArray();
        int count = 0;

        for (int i = 0; i < chars.length; i++) {
            if (chars[i] == '+' || chars[i] == '-' || Character.isDigit(chars[i])) {
                chars[count] = chars[i];
                count++;
                for (int j = i + 1; j < chars.length; j++) {
                    if (Character.isDigit(chars[j])) {
                        chars[count] = chars[j];
                        count++;
                    } else {
                        break;
                    }
                }
                break;
            } else if (chars[i] != ' ') {
                break;
            }
        }

        if (count == 0) {
            return 0;
        }
        if (count == 1 && (chars[0] == '+' || chars[0] == '-')) {
            return 0;
        }
        boolean flag = chars[0] != '-';

        int result = 0;
        int index = 0;
        while (index < count) {
            if (chars[index] == '+' || chars[index] == '-') {
                index++;
                continue;
            }
            int value = chars[index] - '0';

            if (Integer.MAX_VALUE / 10 < result) {
                return flag ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            } else if (Integer.MAX_VALUE / 10 == result) {
                if (flag && value >= 7) {
                    return Integer.MAX_VALUE;
                } else if (!flag && value >= 8) {
                    return Integer.MIN_VALUE;
                }
            }

            result = result * 10 + value;
            index++;
        }
        return flag ? result : -1 * result;
    }

    public int myAtoi2(String str) {
        int res = 0;
        int index = 0;
        while (index < str.length() && str.charAt(index) == ' ') {
            index++;
        }
        if (index == str.length()) {
            return 0;
        }
        int sign = 1;
        if (index < str.length()) {
            if (str.charAt(index) == '+') {
                index++;
            } else if (str.charAt(index) == '-') {
                sign = -1;
                index++;
            }
        }
        while (index < str.length()) {
            char c = str.charAt(index);
            if (c > '9' || c < '0') {
                break;
            }
            if (sign > 0 && (Integer.MAX_VALUE / 10 < res || (Integer.MAX_VALUE / 10 == res && c - '0' > 7))) {
                return Integer.MAX_VALUE;
            }
            if (sign < 0 && (Integer.MAX_VALUE / 10 < res || (Integer.MAX_VALUE / 10 == res && c - '0' > 8))) {
                return Integer.MIN_VALUE;
            }
            res = res * 10 + c - '0';
            index++;
        }
        return res * sign;
    }
}
