import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Solution3Test {

    private static final Solution3 instance = new Solution3();

    @Test
    void findMaxForm() {
        Solution3 solution3 = new Solution3();
        assertEquals(4, solution3.findMaxFormBest(new String[]{"10", "0001", "111001", "1", "0"}, 5, 3));
        assertEquals(2, solution3.findMaxFormBest(new String[]{"10", "0", "1"}, 1, 1));
    }

    @Test
    void testBitSet() {
        Solution3.Bitset bitset = new Solution3.Bitset(2);
        bitset.flip();
        bitset.unfix(1);
        System.out.println(bitset.all());
        bitset.fix(1);
        bitset.fix(1);
        bitset.unfix(1);
        System.out.println(bitset.all());
    }

    @Test
    void search() {
        assertEquals(0, instance.search(new int[]{5, 1, 3}, 5));
    }

    @Test
    void findMin() {
        assertEquals(0, instance.findMin(new int[]{4, 5, 6, 7, 0, 1, 2}));
    }

    @Test
    void numDecodings() {
        assertEquals(3, instance.numDecodings("226"));
        assertEquals(1, instance.numDecodings("10"));
        assertEquals(0, instance.numDecodings("010"));
        assertEquals(0, instance.numDecodings("10011"));
    }

    @Test
    void maximalSquare() {
        assertEquals(4, instance.maximalSquare(new char[][]{
                {'0','1','1','0','1'},{'1','1','0','1','0'},{'0','1','1','1','0'},{'1','1','1','1','0'},{'1','1','1','1','1'},{'0','0','0','0','0'}
        }));
    }
}