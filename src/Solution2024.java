

public class Solution2024 {
    public int convertInteger(int A, int B) {
        if (A == B) {
            return 0;
        }
//        0b11101
//        0b01111
        int xor = A ^ B;
        return Integer.bitCount(xor);
    }
}
