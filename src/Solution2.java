import java.util.HashSet;
import java.util.Set;

public class Solution2 extends SolutionBase {

    //给定一个含有 n 个正整数的数组和一个正整数 target 。
    //找出该数组中满足其和 ≥ target 的长度最小的 连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0 。
    public int minSubArrayLen(int target, int[] nums) {
        int start = 0;
        int end = 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        while (end < nums.length) {
            sum += nums[end];
            while (sum >= target) {
                //记录满足条件的result
                res = Math.min(res, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return res == Integer.MAX_VALUE ? 0 : res;

    }

    //给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续子数组的个数。
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int start = 0;
        int end = 0;
        int prod = 1;
        int res = 0;
        while (end < nums.length) {
            prod *= nums[end];
            //边界start=end，prod=1
            while (start <= end && prod >= k) {
                prod /= nums[start];
                start++;
            }
            res += end - start + 1;
            end++;
        }
        return res;

    }

    //给定一个字符串 s ，请你找出其中不含有重复字符的 最长连续子字符串 的长度。
    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Set<Character> set = new HashSet<>();
        int start = 0, end = 0;
        final int len = s.length();
        while (start < len) {
            if (start != 0) {
                //新一轮遍历，去掉之前的字符
                set.remove(s.charAt(start - 1));
            }
            while (end < len && !set.contains(s.charAt(end))) {
                set.add(s.charAt(end));
                end++;
            }
            //end后++了，因此为实际最后一个不重复的字符索引(end')+1，长度为end'- start + 1 = end - start
            res = Math.max(res, end - start++);
        }
        return res;

    }

    public static void main(String[] args) {
        final Solution2 solution = new Solution2();
        solution.numSubarrayProductLessThanK(new int[]{10}, 2);
    }

}
