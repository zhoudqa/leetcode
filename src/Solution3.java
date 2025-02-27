import annotations.algorithm.Backtrack;
import annotations.algorithm.BinarySearch;
import annotations.level.Medium;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution3 extends SolutionBase {

    //给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
    //
    //请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
    //
    //如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
    public int findMaxForm(String[] strs, int m, int n) {
        //前i个字符串在最多j个0k个1时的最大子集数量, i=0时，dp=0
        int[][][] dp = new int[strs.length + 1][m + 1][n + 1];
        for (int i = 1; i <= strs.length; i++) {
            int zeroCount = (int) strs[i - 1].chars().filter(c -> c == '0').count();
            int oneCount = strs[i - 1].length() - zeroCount;
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    if (zeroCount > j || oneCount > k) {
                        //选不了strs[i]
                        dp[i][j][k] = dp[i - 1][j][k];
                    } else {
                        //选了和不选中max的
                        dp[i][j][k] = Math.max(dp[i - 1][j][k], dp[i - 1][j - zeroCount][k - oneCount] + 1);
                    }
                }
            }
            //不满足条件的情况
        }
        return dp[strs.length][m][n];
    }

    public int findMaxFormBest(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (String str : strs) {
            int zeroCount = (int) str.chars().filter(c -> c == '0').count();
            int oneCount = str.length() - zeroCount;
            for (int j = m; j >= zeroCount; j--) {
                for (int k = n; k >= oneCount; k--) {
                    dp[j][k] = Math.max(dp[j][k], dp[j - zeroCount][k - oneCount] + 1);
                }
            }
        }
        return dp[m][n];
    }

    @Medium
    static class Bitset {

        char[] val;
        char[] reversedVal;
        int len;
        int oneCount;
        int zeroCount;

        public Bitset(int size) {
            zeroCount = size;
            len = size;
            val = new char[size];
            reversedVal = new char[size];
            for (int i = 0; i < size; i++) {
                val[i] = '0';
                reversedVal[i] = '1';
            }
        }

        public void fix(int idx) {
            if (val[idx] == '0') {
                val[idx] = '1';
                reversedVal[idx] = '0';
                oneCount++;
                zeroCount--;
            }
        }

        public void unfix(int idx) {
            if (val[idx] == '1') {
                val[idx] = '0';
                reversedVal[idx] = '1';
                zeroCount++;
                oneCount--;
            }
        }

        public void flip() {
            char[] midVal = reversedVal;
            reversedVal = val;
            val = midVal;
            int mid = zeroCount;
            zeroCount = oneCount;
            oneCount = mid;
        }

        public boolean all() {
            return oneCount == len;
        }

        public boolean one() {
            return oneCount > 0;
        }

        public int count() {
            return oneCount;
        }

        public String toString() {
            return new String(val);
        }
    }

    @Medium
    //给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
    //
    // 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
    //
    //
    //
    //
    //
    // 示例 1：
    //
    //
    //输入：digits = "23"
    //输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
    //
    //
    // 示例 2：
    //
    //
    //输入：digits = ""
    //输出：[]
    //
    //
    // 示例 3：
    //
    //
    //输入：digits = "2"
    //输出：["a","b","c"]
    //
    //
    //
    //
    // 提示：
    //
    //
    // 0 <= digits.length <= 4
    // digits[i] 是范围 ['2', '9'] 的一个数字。
    //
    //
    // Related Topics 哈希表 字符串 回溯 👍 2944 👎 0

    private Map<Character, List<Character>> map = new HashMap<>();


    @Medium
    @Backtrack
    public List<String> letterCombinations(String digits) {
        ArrayList<String> rst = new ArrayList<>();
        if (digits.isEmpty()) {
            return rst;
        }
        map.put('2', Arrays.asList('a', 'b', 'c'));
        map.put('3', Arrays.asList('d', 'e', 'f'));
        map.put('4', Arrays.asList('g', 'h', 'i'));
        map.put('5', Arrays.asList('j', 'k', 'l'));
        map.put('6', Arrays.asList('m', 'n', 'o'));
        map.put('7', Arrays.asList('p', 'q', 'r', 's'));
        map.put('8', Arrays.asList('t', 'u', 'v'));
        map.put('9', Arrays.asList('w', 'x', 'y', 'z'));
        backtrack(rst, digits, 0, new StringBuilder());
        return rst;
    }

    public void backtrack(List<String> rst, String digits, int index, StringBuilder builder) {
        if (index == digits.length()) {
            rst.add(builder.toString());
        } else {
            char n = digits.charAt(index);
            List<Character> currentChars = map.get(n);
            for (Character c : currentChars) {
                builder.append(c);
                backtrack(rst, digits, index + 1, builder);
                builder.deleteCharAt(builder.length() - 1);
            }
        }
    }

    //153: 寻找旋转数组的最小值
    public int findMin(int[] nums) {
        int n = nums.length - 1;
        //只有刚好旋转0次或者长度的整数倍，才会和旋转前顺序一致
        if (nums[0] < nums[n]) {
            return nums[0];
        }
        int l = 0;
        int r = n;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > nums[n]) {
                //右边无序，最小值只能在无序这边
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return nums[r];
    }

    /**
     * 整数数组 nums 按升序排列，数组中的值 互不相同 。
     * <p>
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
     * <p>
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
     * <p>
     * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     */
    @Medium
    @BinarySearch
    public int search(int[] nums, int target) {
        int len = nums.length;
        if (len == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            int midVal = nums[mid];
            if (midVal == target) {
                return mid;
            }
            //分成[left,mid-1] 和 [mid+1,right]
            //升序数组左边比右边大的话说明那边旋转过，那另一边就是有序的
            if (nums[mid] > nums[right]) {
                if (target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[left] > nums[mid]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                //正常二分
                if (target > midVal) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    //https://leetcode.cn/problems/decode-ways/
    // 数字解码字母组合数，[1->'A',2->'B'...26->'Z']
    public int numDecodings(String s) {
        int n = s.length();
        int mod = 3;
        int[] f = new int[mod];
        s = " " + s;
        f[0] = 1;
        //转移方程 f[i] = f[i-1]+f[i-2],a∈[1,9],b∈[10,26];f[i]=f[i-2],b∈[10,26];f[i]=f[i-1],a∈[1,9]
        for (int i = 1; i < n + 1; i++) {
            f[i % mod] = 0;
            int cur = s.charAt(i) - '0';
            int preAndCur = (s.charAt(i - 1) - '0') * 10 + cur;
            boolean aLegal = 1 <= cur && cur <= 9;
            boolean bLegal = 10 <= preAndCur && preAndCur <= 26;
            if (aLegal && bLegal) {
                f[i % mod] = f[(i - 1) % mod] + f[(i - 2) % mod];
            } else if (bLegal) {
                f[i % mod] = f[(i - 2) % mod];
            } else if (aLegal) {
                f[i % mod] = f[(i - 1) % mod];
            }

        }
        return f[n % mod];
    }

    //https://leetcode.cn/problems/maximal-square/description/
    //在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
    public int maximalSquare(char[][] matrix) {
        //f[i][j] = g[i][j]=='0'?0:(1+min(f[i-1,j],f[i,j-1])
        int m = matrix.length;
        int n = matrix[0].length;
        int[] f = new int[n];
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, f[i] = matrix[0][i] == '0' ? 0 : 1);
        }
        for (int i = 1; i < m; i++) {
            int left = matrix[i][0] == '0' ? 0 : 1;
            ans = Math.max(ans, left);
            for (int j = 1; j < n; j++) {
                int cur;
                if (matrix[i][j] == '0') {
                    cur = 0;
                } else {
                    cur = 1;
                    int minLen = Math.min(left, f[j]);
                    if (minLen > 0) {
                        if (matrix[i - minLen][j - minLen] == '1') {
                            cur += minLen;
                        } else {
                            cur += minLen - 1;
                        }
                    }
                }
                f[j] = cur;
                left = cur;
                ans = Math.max(ans, cur);

            }
        }
        return ans * ans;
    }
}
