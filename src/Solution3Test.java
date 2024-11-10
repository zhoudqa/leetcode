import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Solution3Test {

    private static final Solution3 instance = new Solution3();

    @Test
    void findMaxForm() {
        Solution3 solution3 = new Solution3();
        Assertions.assertEquals(4, solution3.findMaxFormBest(new String[]{"10", "0001", "111001", "1", "0"}, 5, 3));
        Assertions.assertEquals(2, solution3.findMaxFormBest(new String[]{"10", "0", "1"}, 1, 1));
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
        Assertions.assertEquals(0, instance.search(new int[]{5, 1, 3}, 5));
    }

    @Test
    void findMin() {
        Assertions.assertEquals(0, instance.findMin(new int[]{4, 5, 6, 7, 0, 1, 2}));
    }
}