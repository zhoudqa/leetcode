import org.junit.jupiter.api.Assertions;

import static org.junit.jupiter.api.Assertions.*;

class Solution2024Test {

    private Solution2024 s = new Solution2024();

    @org.junit.jupiter.api.Test
    void convertInteger() {
        Assertions.assertEquals(2, s.convertInteger(29, 15));
    }
}