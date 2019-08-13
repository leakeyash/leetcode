package array;

public class XofaKindinaDeckofCards {
    public boolean hasGroupsSizeX(int[] deck) {
        int[] count = new int[10000];
        for (int c: deck) {
            count[c]++;
        }

        int g = -1;
        for (int i = 0; i < 10000; ++i) {
            if (count[i] > 0) {
                if (g == -1) {
                    g = count[i];
                } else {
                    g = gcd(g, count[i]);
                }
            }
        }

        return g >= 2;
    }

    private int gcd(int m, int n){
        int temp = 1;
        while(temp!=0){
            temp = m%n;
            m = n;
            n = temp;
        }
        return m;
    }
}
