import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Solution2024Test {

    private Solution2024 s = new Solution2024();

    @org.junit.jupiter.api.Test
    void convertInteger() {
        assertEquals(2, s.convertInteger(29, 15));
    }

    @Test
    void trainingPlan() {
//        assertEquals(1, s.trainingPlan(new int[]{12, 1, 6, 12, 6, 12, 6}));
//        assertEquals(7, s.trainingPlan(new int[]{5, 7, 5, 5}));
        assertEquals(3, s.trainingPlan(new int[]{86,38,67,65,61,72,42,1,17,88,65,72,64,54,22,45,92,1,38,17,3,68,34,64,29,27,6,22,54,56,34,61,38,92,48,82,73,62,86,27,11,6,22,98,86,37,15,61,88,29,73,15,62,1,6,67,11,72,16,87,67,62,42,16,45,98,7,27,87,37,16,56,88,64,15,68,42,98,29,82,65,82,54,7,17,68,92,45,37,87,56,11,48,34,7,48,73}));

    }
}