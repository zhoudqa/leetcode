import java.util.*;
import java.util.function.Consumer;

public class Solution {

    static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    private static ListNode fromArray(int[] source) {
        ListNode next = null;
        ListNode head = null;
        for (int i = source.length - 1; i >= 0; i--) {
            ListNode listNode = new ListNode(source[i]);
            listNode.next = next;
            next = listNode;
            if (i == 0) {
                head = listNode;
            }
        }
        return head;
    }

    private static ListNode fromArray(List<Integer> source) {
        ListNode next = null;
        ListNode head = null;
        for (int i = source.size() - 1; i >= 0; i--) {
            ListNode listNode = new ListNode(source.get(i));
            listNode.next = next;
            next = listNode;
            if (i == 0) {
                head = listNode;
            }
        }
        return head;
    }

    //给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
    //
    //你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");

    }

    // 三数之和为0
    public List<List<Integer>> threeSum(int[] nums) {
        int length = nums.length;
        if (length < 3) {
            return Collections.emptyList();
        }
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            //a不重复
            if (i <= 0 || nums[i] != nums[i - 1]) {
                int twoSum = -nums[i];
                int j = i + 1;
                int k = length - 1;
                //双指针
                while (j < k) {
                    //b不重复 c一定不会重复 假如有一组成功 j必定加1
                    if (j - 1 != i && nums[j] == nums[j - 1]) {
                        j++;
                        continue;
                    }
                    if (nums[j] + nums[k] == twoSum) {
                        res.add(Arrays.asList(nums[i], nums[j++], nums[k--]));
                    } else if (nums[j] + nums[k] < twoSum) {
                        j++;
                    } else k--;
                }
            }
        }
        return res;
    }

    //给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
    //
    //如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
    //
    //您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int over = 0;
        List<Integer> result = new ArrayList<>();
        while (l1 != null || l2 != null || over != 0) {
            int sum = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + over;
            result.add(sum % 10);
            over = sum / 10;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return fromArray(result);
    }

    public int lengthOfLongestSubstringLeetCode(String s) {
        // 哈希集合，记录每个字符是否出现过
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                // 不断地移动右指针
                occ.add(s.charAt(rk + 1));
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }

    //给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
    public int lengthOfLongestSubstring(String s) {
        Set<Integer> largestNoRepeatCharSet = new HashSet<>();


        String longestSubstring = longestSubstringFromIndex(s, 0);
        for (int i = 1; i < s.length(); i++) {
            String longestSubstringFromIndex = longestSubstringFromIndex(s, i);
            if (longestSubstringFromIndex.length() > longestSubstring.length()) {
                longestSubstring = longestSubstringFromIndex;
            }
        }
        return longestSubstring.length();

    }

    private String longestSubstringFromIndex(String s, int index) {
        Set<Integer> noRepeatChars = new HashSet<>();
        for (int i = index; i < s.length(); i++) {
            if (!noRepeatChars.add((int) s.charAt(i))) {
                return s.substring(index, i);
            }
        }
        return s.substring(index);
    }

    //给定两个大小为 m 和 n 的正序（从小到大）数组nums1和nums2。
    //请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
    //你可以假设nums1和nums2不会同时为空。
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int[] mergedArray = mergeArray(nums1, nums2);
        if (mergedArray.length % 2 == 0) {
            int r = mergedArray.length / 2;
            return (double) (mergedArray[r - 1] + mergedArray[r]) / 2;
        } else {
            return mergedArray[mergedArray.length / 2];
        }
    }

    //归并排序
    private int[] mergeArray(int[] nums1, int[] nums2) {
        if (nums1 == null) {
            return nums2;
        }
        if (nums2 == null) {
            return nums1;
        }
        int i = 0;
        int j = 0;
        int length = nums1.length + nums2.length;
        int[] mergedArray = new int[length];
        for (int k = 0; k < length; k++) {
            if (i < nums1.length && j < nums2.length) {
                if (nums1[i] < nums2[j]) {
                    mergedArray[k] = nums1[i++];
                } else {
                    mergedArray[k] = nums2[j++];
                }
            } else if (i < nums1.length) {
                mergedArray[k] = nums1[i++];
            } else if (j < nums2.length) {
                mergedArray[k] = nums2[j++];
            }
        }
        return mergedArray;
    }

    //给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
    //dp[m][n] 第m到第n位为回文字符串
    public String longestPalindrome(String s) {
        String result = "";
        int length = s.length();
        boolean[][] dp = new boolean[length][length];
        int maxLen = 0;
        for (int distance = 0; distance < length; distance++) {
            for (int i = 0; i < length - distance; i++) {
                int j = i + distance;
                if (i == j) {
                    dp[i][j] = true;
                } else {
                    boolean b = s.charAt(i) == s.charAt(j);
                    if (j == i + 1) {
                        dp[i][j] = b;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1] && b;
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    result = s.substring(i, j + 1);
                    maxLen = j - i + 1;
                }
            }
        }
        return result;
    }

    //两步问题。有个小孩正在上楼梯，楼梯有 n 阶台阶，小孩一次可以上 1 阶、2 阶。计算小孩有多少种上楼梯的方式。
    public long waysToUpStairs(int n) {
        long[] res = new long[n];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                res[i] = 1;
            } else if (i == 1) {
                res[i] = 2;
            } else {
                res[i] = res[i - 1] + res[i - 2];
            }
        }
        return res[n - 1];
    }

    //将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
    //比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：
    //L   C   I   R
    //E T O E S I I G
    //E   D   H   N
    //之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。
    public String convertZ(String s, int numRows) {
        if (numRows == 1 || s.length() < numRows) {
            return s;
        }
        int mod = numRows * 2 - 2;
        int mid = mod / 2;
        StringBuilder[] ss = new StringBuilder[numRows];
        for (int i = 0; i < s.length(); i += mod) {
            String substring = s.substring(i, Math.min(i + mod, s.length()));
            for (int j = 0; j < substring.length(); j++) {
                int index = Math.abs(mid - j);
                if (ss[index] == null) {
                    ss[index] = new StringBuilder();
                }
                ss[index].append(substring.charAt(j));
            }
        }
        StringBuilder result = new StringBuilder();
        for (int i = ss.length - 1; i >= 0; i--) {
            result.append(ss[i]);
        }
        return result.toString();
    }

    //给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
    public int reverse(int x) {
        if (x == 0) {
            return 0;
        }
        int res = 0;
        boolean isPositive = x > 0;
        int maxMid = Integer.MAX_VALUE / 10;
        x = Math.abs(x);
        while (x > 0) {
            int pop = x % 10;
            if (res > maxMid || res == maxMid && x > (isPositive ? 7 : 8)) {
                return 0;
            }
            res = res * 10 + pop;
            x /= 10;
        }
        return isPositive ? res : -1 * res;
    }

    //请你来实现一个 atoi 函数，使其能将字符串转换成整数。
    //首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。接下来的转化规则如下：
    //
    //如果第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字字符组合起来，形成一个有符号整数。
    //假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成一个整数。
    //该字符串在有效的整数部分之后也可能会存在多余的字符，那么这些字符可以被忽略，它们对函数不应该造成影响。
    //注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换，即无法进行有效转换。
    //
    //在任何情况下，若函数不能进行有效的转换时，请返回 0 。
    //提示：
    //本题中的空白字符只包括空格字符 ' ' 。
    //假设我们的环境只能存储 32 位大小的有符号整数。如果数值超过这个范围，请返回INT_MAX或INT_MIN。
    public int myAtoi(String str) {
        int space = 32;
        int positive = 43;
        int negative = 45;
        byte[] bytes = str.getBytes();
        boolean startCount = false;
        int max = Integer.MAX_VALUE / 10;
        boolean isNegative = false;
        int res = 0;
        for (byte aByte : bytes) {
            if (startCount) {
                if (isNum(aByte)) {
                    int val = aByte - 48;
                    if (res > max || res == max && val > (isNegative ? 8 : 7)) {
                        return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
                    }
                    res = res * 10 + aByte - 48;
                } else {
                    break;
                }
            }
            if (!startCount && aByte != space) {
                startCount = true;
                if (aByte == negative || aByte == positive) {
                    isNegative = aByte == negative;
                } else if (isNum(aByte)) {
                    res = res * 10 + aByte - 48;
                } else {
                    return 0;
                }
            }

        }
        return isNegative ? -1 * res : res;
    }

    private boolean isNum(byte aByte) {
        return aByte >= 48 && aByte <= 57;
    }

    //判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
    public boolean isPalindrome(int x) {
        if (x == 0) {
            return true;
        } else if (x < 0) {
            return false;
        } else {
            return isPalindrome(String.valueOf(x));
        }
    }

    private boolean isPalindrome(String positiveNum) {
        if (positiveNum.length() < 2) {
            return true;
        } else {
            return positiveNum.charAt(0) == positiveNum.charAt(positiveNum.length() - 1) && isPalindrome(positiveNum.substring(1, positiveNum.length() - 1));
        }

    }

    //给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
    //
    //'.' 匹配任意单个字符
    //'*' 匹配零个或多个前面的那一个元素
    // f(m,n)表示s的前m个字符和前n个字符可以匹配
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }

    //s的第i个字符和p的第j个字符匹配
    private boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    //给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。
    // 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    //
    //说明：你不能倾斜容器。
    public int maxArea(int[] height) {
        int i = 0;
        int j = height.length - 1;
        int max = (j - i) * Math.min(height[i], height[j]);
        while (i < j) {
            //移动矮的指针
            if (height[j] < height[i]) {
                j--;
            } else {
                i++;
            }
            int currentArea = (j - i) * Math.min(height[i], height[j]);
            max = Math.max(max, currentArea);
        }
        return max;
    }

    //排序
    //common functions
    private static <T extends Comparable<T>> boolean less(T a, T b) {
        return a.compareTo(b) < 0;
    }

    private static <T extends Comparable<T>> void exch(T[] a, int i, int j) {
        if (i == j) return;
        T t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    private static <T extends Comparable<T>> boolean sorted(T[] a) {
        for (int i = 1; i < a.length; i++) {
            if (less(a[i], a[i - 1])) return false;
        }
        return true;
    }

    private static class SimpleComparableVal implements Comparable<SimpleComparableVal> {
        int val;

        public SimpleComparableVal(int val) {
            this.val = val;
        }

        @Override
        public int compareTo(SimpleComparableVal o) {
            return this.val - o.val;
        }

        @Override
        public String toString() {
            return "" + val;
        }
    }

    private static SimpleComparableVal[] toComparableArray(int[] source) {
        SimpleComparableVal[] destination = new SimpleComparableVal[source.length];
        for (int i = 0; i < source.length; i++) {
            destination[i] = new SimpleComparableVal(source[i]);
        }
        return destination;
    }

    public static <T extends Comparable<T>> void selectionSort(T[] a) {
        int length = a.length;
        for (int i = 0; i < length; i++) {
            int minIndex = i;
            for (int j = i + 1; j < length; j++) {
                if (less(a[j], a[minIndex])) {
                    minIndex = j;
                }
            }
            exch(a, minIndex, i);
        }
    }

    public static <T extends Comparable<T>> void insertionSort(T[] a) {
        for (int i = 1; i < a.length; i++) {
            // 将a[i]和a[i-1]...a[0]中比a[i]大的交换，直到遇到比a[i]小的
            for (int j = i; j > 0 && less(a[j], a[j - 1]); j--) {
                exch(a, j, j - 1);
            }
        }
    }

    public static <T extends Comparable<T>> void shellSort(T[] a) {
        int h = 1;
        while (h < a.length / 3) {
            h = 3 * h + 1;
        }
        while (h >= 1) {
            for (int i = h; i < a.length; i++) {
                for (int j = i; j >= h && less(a[j], a[j - h]); j -= h) {
                    exch(a, j, j - h);
                }
            }
            h /= 3;
        }
    }

    @SuppressWarnings("unchecked")
    public static <T extends Comparable<T>> void merge(T[] a, T[] temp, int lo, int hi, int mid) {
        int i = lo;
        int j = mid + 1;
        int tIndex = 0;
        while (i <= mid && j <= hi) {
            if (less(temp[i], temp[j])) temp[tIndex++] = a[i++];
            else temp[tIndex++] = a[j++];
        }
        while (i <= mid) {
            temp[tIndex++] = a[i++];
        }
        while (j <= hi) {
            temp[tIndex++] = a[j++];
        }
        tIndex = 0;
        while (lo <= hi) {
            a[lo++] = temp[tIndex++];
        }
    }

    public static <T extends Comparable<T>> void mergeSort(T[] a) {
//        Comparable[] temp = new Comparable[a.length];
//        mergeSort(a, temp, 0, a.length - 1);
    }

    public static <T extends Comparable<T>> void mergeSort(T[] a, T[] temp, int lo, int hi) {
        if (hi <= lo) return;
        int mid = (hi + lo) / 2;
        mergeSort(a, temp, lo, mid);
        mergeSort(a, temp, mid + 1, hi);
        merge(a, temp, lo, hi, mid);
    }

    //根节点到叶节点的所有路径
    public static void printTree(TreeNode root) {

        List<String> res = new ArrayList<>();
        if (root != null)
            printTree(root, "", res);
        res.forEach(s -> System.out.println(reverse(s)));

    }

    public static void printTree(TreeNode root, String cur, List<String> res) {
        if (root == null) {
            return;
        }
        cur += root.val;
        if (root.left == null && root.right == null) {
            res.add(cur);
        } else {
            printTree(root.right, cur, res);
            printTree(root.left, cur, res);
        }

    }

    private static String reverse(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = s.length() - 1; i >= 0; i--) {
            stringBuilder.append(s.charAt(i));
        }
        return stringBuilder.toString();
    }


    //整数转罗马字符串
    public String intToRoman(int num) {
        String res = "";
//                                 1000 500  100  50   10    5    1
        char[] basics = new char[]{'M', 'D', 'C', 'L', 'X', 'V', 'I'};

        int f = 3;
        int charIndex = 0;
        while (f >= 0) {
            int carry = (int) Math.pow(10, f);
            int cn = num / carry;
            if (cn != 0) {
                char curChar = basics[charIndex];
                if (cn == 9) {
                    res += curChar;
                    res += basics[charIndex - 2];
                } else {
                    if (cn == 4) {
                        res += curChar;
                        res += basics[charIndex - 1];
                    } else if (cn >= 5) {
                        res += basics[charIndex - 1];
                        appendNTimes(res, cn - 5, curChar);
                    } else {
                        appendNTimes(res, cn, curChar);
                    }
                }
            }
            charIndex += 2;
            num %= carry;
            f--;
        }
        return res;

    }

    public void appendNTimes(String builder, int n, char c) {
        if (n == 0) {
            return;
        }
        for (int i = 0; i < n; i++) {
            builder += c;
        }
    }

    public int romanToInt(String s) {

        int res = 0;
        int i = 0;
        int len = s.length();
        while (i < len) {
            if (i + 1 < len) {
                res += getValue(s.charAt(i)) < getValue(s.charAt(i + 1)) ? -getValue(s.charAt(i)) : getValue(s.charAt(i));
            } else {
                res += getValue(s.charAt(i));
            }
            i++;
        }
        return res;

    }

    private int getValue(char ch) {
        switch (ch) {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;
        }
    }

    public String longestDupSubstring(String s) {
        List<String> suffixSubStrs = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            suffixSubStrs.add(s.substring(i));
        }
        Collections.sort(suffixSubStrs);
        int maxLen = 0;
        String res = "";
        for (int i = 1; i < suffixSubStrs.size(); i++) {
            String longestDupString = longestDupString(suffixSubStrs.get(i), suffixSubStrs.get(i - 1));
            if (maxLen < longestDupString.length()) {
                maxLen = longestDupString.length();
                res = longestDupString;
            }
        }
        return res;

    }

    private String longestDupString(String s1, String s2) {
        for (int i = 0; i < s1.length(); i++) {
            if (i >= s2.length() || s1.charAt(i) != s2.charAt(i)) {
                return s1.substring(0, i);
            }
        }
        return s1;
    }


    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftMinDep = minDepth(root.left);
        int rightMinDep = minDepth(root.right);
        return 1 + (leftMinDep == 0 ? rightMinDep : rightMinDep == 0 ? leftMinDep : Math.min(leftMinDep, rightMinDep));

    }

    public boolean hasPathSum(TreeNode root, int targetSum) {

        if (root == null) {
            return false;
        }

        if (root.left == null && root.right == null) {
            return root.val == targetSum;
        } else
            return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);

    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tempRight = root.right;
        root.right = root.left != null ? invertTree(root.left) : null;
        root.left = tempRight != null ? invertTree(tempRight) : null;
        return root;
    }

    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        String cur = "";
        return sumNb(root, cur);
    }

    public int sumNb(TreeNode index, String cur) {
        cur += index.val;
        if (index.left == null && index.right == null) {
            return Integer.parseInt(cur);
        } else if (index.left == null) {
            return sumNb(index.right, cur);
        } else if (index.right == null) {
            return sumNb(index.left, cur);
        } else return sumNb(index.left, cur) + sumNb(index.right, cur);
    }

    //找寻p q节点的公共最小祖先
    //思路：如果当前节点=P或Q那么P或Q最小祖先是自己 否则在左右子树寻找P或Q最小公共祖先，左右都找到了 那么当前是最小公共祖先，否则P和Q都在左边或者右边
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = this.lowestCommonAncestor(root.left, p, q);
        TreeNode right = this.lowestCommonAncestor(root.right, p, q);
        return left != null && right != null ? root : left != null ? left : right;
    }

    //前序头结点即为根节点，按根节点split中序 当前前序为[root]+[左子树前序]+[右子树前序] 当前中序为[左子树中序]+[root]+[右子树中序]
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0) {
            return null;
        }
        TreeNode rootNode = new TreeNode(preorder[0], null, null);
        if (preorder.length == 1) {
            return rootNode;
        }
        int rootInIndex = findIndexOf(inorder, preorder[0]);
        rootNode.left = buildTree(Arrays.copyOfRange(preorder, 1, rootInIndex + 1), Arrays.copyOfRange(inorder, 0, rootInIndex));
        rootNode.right = buildTree(Arrays.copyOfRange(preorder, rootInIndex + 1, preorder.length), Arrays.copyOfRange(inorder, rootInIndex + 1, inorder.length));
        return rootNode;
    }

    private int findIndexOf(int[] inorder, int i) {
        for (int j = 0; j < inorder.length; j++) {
            if (i == inorder[j]) {
                return j;
            }
        }
        throw new RuntimeException("NOT FOUND");
    }

    public TreeNode recoverFromPreorder(String S) {
        return recoverFromPreorder(S, 0);
    }

    public TreeNode recoverFromPreorder(String s, int depth) {
        if (s == null || s.length() == 0) {
            return null;
        }
        int firstIndexOfLink = s.indexOf('-');
        if (firstIndexOfLink == -1) {
            //无子节点
            return new TreeNode(Integer.parseInt(s), null, null);
        }
        TreeNode root = new TreeNode(Integer.parseInt(s.substring(0, firstIndexOfLink)), null, null);
        int biggestNextDepthIndex = findBiggestNextDepthIndex(s, depth);
        int leftNodeIndex = firstIndexOfLink + depth + 1;
        if (biggestNextDepthIndex > leftNodeIndex) {
            //有左右子树
            root.left = recoverFromPreorder(s.substring(leftNodeIndex, biggestNextDepthIndex - (depth + 1)), depth + 1);
            root.right = recoverFromPreorder(s.substring(biggestNextDepthIndex), depth + 1);
        } else if (biggestNextDepthIndex > 0) {
            //只有左子树
            root.left = recoverFromPreorder(s.substring(leftNodeIndex), depth + 1);
        }

        return root;
    }

    private int findBiggestNextDepthIndex(String s, int depth) {
        int countLink = 0;
        int result = -1;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '-') {
                countLink++;
            } else {
                if (countLink == depth + 1) {
                    result = i;
                }
                countLink = 0;
            }
        }
        return result;
    }

    //DP
    //最长连续子数组 fn为以nums[n]结尾的最长子数组
    public int maxSubArray(int[] nums) {
        int fn = 0;
        int res = nums[0];
        for (int num : nums) {
            fn = Math.max(fn + num, num);
            res = Math.max(res, fn);
        }
        return res;
    }

    //最长递增子序列长度:fn为nums[0...n]之间的包含nums[n]最长递增子序列长度
    public int lengthOfLIS(int[] nums) {
        int res = 1;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                }
            }
            res = Math.max(dp[i], res);
        }
        return res;
    }

    //最长递增子序列的个数:fn为包含nums[n]的最长递增子序列个数 其长度为len[n] 其个数为count[n]
    public int findNumberOfLIS(int[] nums) {
        int[] len = new int[nums.length];
        int[] count = new int[nums.length];
        len[0] = 1;
        count[0] = 1;
        int maxLen = 1;
        int maxLenCount = 1;
        for (int i = 1; i < nums.length; i++) {
            count[i] = 1;
            len[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (len[i] < len[j] + 1) {
                        //长度可以增长
                        len[i] = len[j] + 1;
                        count[i] = count[j];
                    } else if (len[i] == len[j] + 1) {
                        //长度不增长
                        count[i] += count[j];
                    }
                }
            }
            if (len[i] > maxLen) {
                maxLen = len[i];
                maxLenCount = count[i];
            } else if (len[i] == maxLen) {
                maxLenCount += count[i];
            }
        }
        return maxLenCount;
    }

    //TODO 俄罗斯套娃信封 fn为包含fn-1的最优解
    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes == null || envelopes.length == 0) {
            return 0;
        }
        int[] len = new int[envelopes.length];
        int[] count = new int[envelopes.length];
        len[0] = 1;
        count[0] = 1;
        int maxLen = 1;
        int maxLenCount = 1;
        for (int i = 1; i < envelopes.length; i++) {
            count[i] = 1;
            len[i] = 1;
            for (int j = 0; j < i; j++) {
                if (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1] ||
                        envelopes[i][0] < envelopes[j][0] && envelopes[i][1] < envelopes[j][1]) {
                    if (len[i] < len[j] + 1) {
                        //长度可以增长
                        len[i] = len[j] + 1;
                        count[i] = count[j];
                    } else if (len[i] == len[j] + 1) {
                        //长度不增长
                        count[i] += count[j];
                    }
                }
            }
            if (len[i] > maxLen) {
                maxLen = len[i];
                maxLenCount = count[i];
            } else if (len[i] == maxLen) {
                maxLenCount += count[i];
            }
        }
        return maxLen;
    }

    //买卖股票一次买入一次卖出
    public int maxProfit(int[] prices) {
        if (prices.length < 2) {
            return 0;
        }
        int minPrice = prices[0];
        int res = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (res < prices[i] - minPrice) {
                res = prices[i] - minPrice;
            }
        }
        return res;
    }

    //买卖股票多次买入卖出
    public int maxProfit2(int[] prices) {
        int res = 0;
        for (int i = 1; i < prices.length; i++) {
            res += Math.max(prices[i] - prices[i - 1], 0);
        }
        return res;
    }

    public void dfs(TreeNode root) {
        if (root != null) {
            System.out.print(root.val);
            dfs(root.left);
            dfs(root.right);
        }
    }

    public void dfsPreorderWithoutRecursion(TreeNode root) {
        Stack<TreeNode> temp = new Stack<>();
        temp.push(root);
        while (!temp.empty()) {
            TreeNode pop = temp.pop();
            System.out.print(pop.val);
            if (pop.right != null) {
                temp.push(pop.right);
            }
            if (pop.left != null) {
                temp.push(pop.left);
            }
        }
    }

    public void dfsInorderWithoutRecursion(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> temp = new Stack<>();
            while (!temp.isEmpty() || root != null) {
                if (root != null) {
                    //入栈
                    temp.push(root);
                    root = root.left;
                } else {
                    //无左子节点了 出栈
                    root = temp.pop();
                    System.out.print(root.val);
                    root = root.right;
                }
            }
        }
    }

    public void dfsPostorderWithoutRecursion(TreeNode root) {
        Stack<TreeNode> temp = new Stack<>();
        Stack<TreeNode> temp1 = new Stack<>();
        temp.push(root);
        while (!temp.empty()) {
            TreeNode pop = temp.pop();
            temp1.push(pop);
            if (pop.right != null) {
                temp.push(pop.right);
            }
            if (pop.left != null) {
                temp.push(pop.left);
            }
        }
        while (!temp1.isEmpty()) {
            System.out.print(temp1.pop().val);
        }
    }


    public void bfs(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            System.out.print(cur.val);
            if (cur.left != null) {
                queue.add(cur.left);
            }
            if (cur.right != null) {
                queue.add(cur.right);
            }
        }
    }

    public void bfsWithLevel(TreeNode root) {
        if (root == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            //遍历同一层的节点而不是每个节点都进入while循环，记住当前队列中的节点数量
            int sameLevelNodeCount = queue.size();
            for (int i = 0; i < sameLevelNodeCount; i++) {
                TreeNode cur = queue.poll();
                System.out.print(cur.val);
                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
            }
            System.out.println();
        }
    }

    public List<List<Integer>> zigzagBfsWithLevel(TreeNode root) {
        if (root == null) {
            return Collections.emptyList();
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<List<Integer>> res = new ArrayList<>();
        boolean leftToRight = true;
        while (!queue.isEmpty()) {
            //遍历同一层的节点而不是每个节点都进入while循环，记住当前队列中的节点数量
            int sameLevelNodeCount = queue.size();
            //使用双端队列决定是插在列表头还是尾
            Deque<Integer> sameLevelNodeVals = new LinkedList<>();
            for (int i = 0; i < sameLevelNodeCount; i++) {
                TreeNode cur = queue.poll();
                if (leftToRight) {
                    sameLevelNodeVals.offerFirst(cur.val);
                } else {
                    sameLevelNodeVals.offerLast(cur.val);
                }
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            //每层转换顺序
            leftToRight = !leftToRight;
            res.add(new ArrayList<>(sameLevelNodeVals));
        }
        return res;
    }

    //验证二叉搜索树
    //使用DFS遍历 若是存在逆序返回false
    public boolean isValidBST(TreeNode root) {
        return dfsLMROrdered(root, new ArrayList<>());
    }

    //中序遍历且必须有序
    private boolean dfsLMROrdered(TreeNode node, List<Integer> stack) {
        if (node == null) {
            return true;
        }
        return dfsLMROrdered(node.left, stack) && putIfOrdered(stack, node.val) && dfsLMROrdered(node.right, stack);
    }

    private boolean putIfOrdered(List<Integer> stack, int val) {
        if (stack.isEmpty()) {
            return stack.add(val);
        }
        return val > stack.get(stack.size() - 1) && stack.add(val);
    }

    //给你二叉树的根结点 root ，请你将它展开为一个单链表：
    //
    //展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
    //展开后的单链表应该与二叉树 先序遍历 顺序相同。
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        flattenThenReturnLowestNode(root);
    }

    // 时间O(N) 空间O(1)
    private TreeNode flattenThenReturnLowestNode(TreeNode node) {
        if (node.left == null && node.right == null) {
            return node;
        } else if (node.left != null) {
            TreeNode rightTemp = node.right;
            node.right = node.left;
            TreeNode leftLowestNode = flattenThenReturnLowestNode(node.left);
            node.left = null;
            if (rightTemp != null) {
                leftLowestNode.right = rightTemp;
                return flattenThenReturnLowestNode(rightTemp);
            } else {
                return leftLowestNode;
            }
        } else {
            return flattenThenReturnLowestNode(node.right);
        }
    }

    //有效括号
    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        Map<Character, Character> lrMap = new HashMap<>();
        lrMap.put(']', '[');
        lrMap.put(')', '(');
        lrMap.put('}', '{');
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            //左括号入栈，右括号匹配
            if (lrMap.containsValue(c)) {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                if (stack.peek() != lrMap.get(c)) {
                    return false;
                } else {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
    }

    //合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        if (lists.length == 1) {
            return lists[0];
        } else {
            boolean b = lists.length % 2 == 0;
            ListNode[] splited = new ListNode[b ? lists.length / 2 : lists.length / 2 + 1];
            for (int i = 0; i < splited.length; i++) {
                if (!b && i == splited.length - 1) {
                    splited[i] = lists[lists.length - 1];
                } else {
                    splited[i] = mergeTwoListNodes(lists[i * 2], lists[i * 2 + 1]);
                }
            }
            return mergeKLists(splited);
        }
    }

    //合并两个升序链表
    ListNode mergeTwoListNodes(ListNode a, ListNode b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        ListNode head;
        if (a.val < b.val) {
            head = a;
            a = a.next;
        } else {
            head = b;
            b = b.next;
        }
        ListNode cur = head;
        while (a != null && b != null) {
            if (a.val < b.val) {
                cur.next = a;
                a = a.next;
            } else {
                cur.next = b;
                b = b.next;
            }
            cur = cur.next;
        }
        if (a != null) {
            cur.next = a;
        }
        if (b != null) {
            cur.next = b;
        }
        return head;
    }

    //输入：path = "/a/./b/../../c/"
    //输出："/c"
    //TODO
    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return null;
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < path.length(); i++) {

        }


    }

    //全排列 回溯算法
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfs(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth,
                     Deque<Integer> path, boolean[] used,
                     List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;

                System.out.println("  递归之前 => " + path);
                dfs(nums, len, depth + 1, path, used, res);

                used[i] = false;
                path.removeLast();
                System.out.println("递归之后 => " + path);
            }
        }
    }

    //N皇后 回溯法
    public List<List<String>> solveNQueens(int n) {
        if (n == 0) {
            return Collections.emptyList();
        } else if (n == 1) {
            return Collections.singletonList(Collections.singletonList("Q"));
        }
        List<List<String>> res = new ArrayList<>();
        backtrack(new LinkedList<>(), 0, n, res);
        return res;
    }

    private void backtrack(LinkedList<String> cur, int row, int n, List<List<String>> res) {
        if (row == n) {
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = 0; i < n; i++) {
            if (isValid(cur, i, row, n)) {
                //有效的话填写下一行
                cur.offerLast(appendOneRow(i, n));
                backtrack(cur, row + 1, n, res);
                //状态恢复
                cur.removeLast();
            }
        }
    }

    private boolean isValid(List<String> cur, int pos, int row, int n) {
        //竖向重复|
        for (int i = 0; i < row; i++) {
            if (cur.get(i).charAt(pos) == 'Q') {
                return false;
            }
        }
        //斜向重复\
        for (int i = row - 1, j = pos - 1; i >= 0 && j >= 0; i--, j--) {
            if (cur.get(i).charAt(j) == 'Q') {
                return false;
            }
        }
        //斜向重复/
        for (int i = row - 1, j = pos + 1; i >= 0 && j < n; i--, j++) {
            if (cur.get(i).charAt(j) == 'Q') {
                return false;
            }
        }
        return true;
    }

    public String appendOneRow(int pos, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pos; i++) {
            sb.append(".");
        }
        sb.append("Q");
        for (int i = pos + 1; i < n; i++) {
            sb.append(".");
        }
        return sb.toString();
    }

    //编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值
    public boolean searchMatrix(int[][] matrix, int target) {
        int i = matrix[0].length - 1;
        int j = 0;
        while (i >= 0 && j < matrix.length) {
            int delta = matrix[j][i] - target;
            if (delta == 0) {
                return true;
            } else if (delta < 0) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    //请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。
    //
    //例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        for (int i = 0; i < T.length; i++) {
            res[i] = 0;
            for (int j = i + 1; j < T.length; j++) {
                if (T[j] > T[i]) {
                    res[i] = j - i;
                    break;
                }
            }
        }
        return res;
    }

    //单调栈
    public int[] dailyTemperaturesBest(int[] T) {
        int[] res = new int[T.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < T.length; i++) {
            int temp = T[i];
            while (!stack.isEmpty() && temp > T[stack.peek()]) {
                Integer preIndex = stack.pop();
                res[preIndex] = i - preIndex;
            }
            stack.push(i);
        }
        return res;
    }

    public static void main(String[] args) {
//        int size = 20;
//        int[] a = new int[size];
//        Random random = new Random();
//        int bound = 10000;
//        for (int i = 0; i < size; i++) {
//            a[i] = random.nextInt(bound);
//        }
//        testSortAlgorithm(a, Solution::mergeSort);
//        testSortAlgorithm(a, Solution::shellSort);
//        System.out.println(new Solution().intToRoman(1994));
//        System.out.println(new Solution().romanToInt("VIII"));
//        System.out.println(new Solution().longestDupSubstring("uirqkccmgnpbqwfkmplmfovxkjtgxbhwzqdlydunpuzxgvodnaanxpdiuhbitfllitaufvcmtneqmvibimvxmzvysignmsfykppdktpfhhcayfcwcrtjsgnyvzytevlqfabpjwxvjtfksztmdtrzgagdjtpeozzxppnkdbjayfbxqrqaefkcdbuvteobztucgdbcmlgnxldcwtzkvhwqxbsqlbsrbvesemhnbswssqvbiketdewfauvtzmyzrrqslzagjcyuznkpkgpkkinldxcjtuoumcbcttabfuzjbtoqjqbpnsemojbtctvdmobnuutcsfmhkrbwkmpdcdncqizgtlqekvqophqxewkpxpkrgjivrtarxtwbbfhfikbrdwohppyiyyztopksdmxvqiaafswyjfnlwntldwniypzaxrscyqfrlqehqgzaxxazfwtwjajsrmwuerdqklhwoqxptcvqoqbjwfqrewirtcbskmkaqgtcnxnsqmwgwjxruhyyjtbuivvepnxiwjmtlrvexjzevctflajibxkvmbzdfwoqobmhstgpshtxitwttpkdvfmfwtwsazfgzwbtmqrowrcesyyeuwunodesrzbmjbxnchaqptfgqlknuarhgnsbwnucdhckpbwhtwhejivrmuccbwnyenyvonquscneswngwbkjysxvdwbzymwxcrnexrxhmuwvycmsiazmqavgmyurbcmvdjplloidbzacunerwobvaxsromiiwzqxnrsjpyoacfxcmmogmokhpmhxzkdzmpjcrgaeihdhczrmxmfurjatuwxriiwtfojwvkkybcdmwayhnzrnqrynwtrvmtgtrxndlbtlhyzfjtbtvqujjuwpibuonuwjdfvnhdqqzlmwwheztjkrrzrroogovapywxkjsccjnseanhxijybintgbjwlkmdzuoeclfqatffgjvcbujovunnauprhoocxzghzvsmuyhslnwqlcsdprwutyqggxfvczqiqeatglhubmllcvcqxrvcojllcoufzfxlunpcpfmypcjbnhtnhgyxriupmpvioqnkibldpnjxctyycwdxloucneypbxwiehqmzkvmwymuurykodpvylicnokwwencbbvzviljapeabhjagrzrmvgebwyrmctksubnhawahjclnpmavhsirvewqbuymdwxcsxtcwehreejtwoagewkmyhwzvyurouyffwhnsyhpiiozetsoveeqndctcxpghtxkpaolqjfaiwdtfjxkvsvamtmhyzuqttiaqhbkzxdjoqxnwpqnykcfaoteretoektileadlqizzlfgcwyhtgyjxdznvaohkgukmhjspqxgdygnulmmrwpwpkvlesvkqtuihzomyxedvalptrineagfdeotlvapabqrebqszkdzekxvttytwfzlsmzyfihcrvsirociunhuplyaukcoiyogdbtooijyfazmfpqwzfgdqzzcggnytspjihxeekoiafknmdvmeppdgsbyeqlpxhjkibcrhknfkznuqanixzckkcufqkvfbcmvrmboqkfablshfdcqcuyjqybsgzpdnjcpzekrydnoxjpwyttblbcadjifmrpexjvunecxnkgeusxwqxujanhdfjuaijngssxxgldxckiucpclhmzcwdhpvdothkdkwnopugrxffhqwwxtxvbralbzpzsgtkdplydydrlckyzzsxypywnavgnkxdbgptkeagvdhigxbnksioopfpplizvztoqrpwkyzazqvwxirablkyvyxfvemgtyrwubpdsuwzpahsnjepppwadoqeawrzhgrvuxexrqikkfpcihlypsnuhqnwzjfkxbrszsgzdmoomlojemujdwhpdinhmwaagbgovhziwrvpwszsvobdxbccxonqilhggfvnejiixcitcpisomwchguxhchizeyxtshuyqmjlhnrvhbbqgultmndqbojwqwfzqzlwvqazixjrivjcignlfhqfrmqhxpcarwbwwkepecdezudavkfkfdefzrrovazszvexdjtmqdaazzqhydakrvhihnmmetpdqalkhlkaavvpolsfzehmhpslrbscywgybctiunwxbqkojxjmyixqfoxwkpowukicsdgkonbqqqgawqgcweywlyntjonqfrzzlnefktbfrutbeqsjktlvvinrpcnorybwuocjothvohrnjeatbkuluguhazkhfruhobjtbnkfqqawftrrrmckzopldnhmwdrzbxdfbvbnmpxnymipabizlmjfmufzsjvazutsvvxdytuiwgmredaceyoyrjtxaikgtxnirdfwaffydnufzjvfsotsnvtpuueygzicyvvdnqmqsgbdbutwydtadejpjjybguekluohpmmazslvyrlqjtellkzbfvethnsoayrjhmtjywkczieszgzrncxavcdpovmqwofeaqeksfonmdjfcdqwzecgjaciriymkklabtmklircuomrdpeqtgkwlyqbzxditppvyhdnajwfcuedtxlmdkhhaggjolunfpojakvgkamiwccghznakxwvqgpfctbeadvcuwijlaitmubgqkbazlrutvlsahzhzxhydedramruudkqfqeycvyrclkuomsyyqarlwpmoqvtqlflscjrpctscizelgffwkhlftpihqgnuvkdfqkafcufpkhosbtdeycfhwbhjiwqbwpwicwywafqvazkjiibtohdbaqhvfdntufauxcsjkgaobbyxmgfoqxggnhwpifegjehpgnnoshkjbwyleevrnzfzxrhlmlittjcxjjaboyuepufctqfyiaoalxcindpluunrxvnegslpigqofsqozkxigbgokkcpougplkqdpermnnpgelgxogrbiufvzphzhnrswkkcuscnuxrfenhjhwjbzmydfnughyifenxtlqpeexmsuzuzgbfebbdyrirzvgrvdmlagbbycqqwsxzeqyjokccevcebniyxzznxqkqrvizkncfxbmpabsigucewylsghvthkauqddarvfuoxpxpowtflhxxspbsncyekwtiywmyakrnxqphdxkwwlmvdilqsjkraemegbslpkzrfggebukefcqwsliqjwxkpnjzmqchicyqnvacismzgqncurzuqpwwpjnpsgtgxszraeavhopzcrsbvemrwxjvctjwjrjjkjlesmypjxnoidprodufuskpuphcaudfqhyyhlczztqqrwlqvzsgmqnwizrgnuzaxqtekjdznxxxbhubszwjxvynpyifllqjvxebxyfqvwuojitusnbutylahvzinydvqvlhezxkbemtrklswlkmipoidiiwilqrdamjreetphgexhrhflyevdmxetsctrhorjesmuzdgtdirwqftgvomtbjvwymubuajfvevecruuppxhyotbdfhzvtiwewaiypqublanrwotmehnhdojgmlslifsgetpnaetavkxujboxakttvgrkadnwzhslaryrvbedtrahddsnpuapxruqvvctrjtsskziknuonrglqcmtpwtdaakhlkxneqxkchrfkpwznektfadgwetfhwnmzupkerdouulifjmsxvamgeumgigdldtpeklgmlppjizsocvljmuckqtotdbhhavclarnifecmqmjomotpathuyisjpbnkzgyslvnsxctuzwwdyxikjvfyizqnvqovtenxsxybateirdeunrtknnhiwfypfscisinkwjnassykklybrfdsrapuebvjpshezxctfydulllekuikdbndjnessrrknotidmsdfertiertauzfspljdvnuybmjideoutgimgcecyqxgkfglgfkkytgqoznrizealqpzfzrahsxcyngkiynhggpxietbgynflyufwchqdlcmrvfdhqkbmyctsikvnugswgdmvfacbvnzdptroagieqqwaxkojcipzwvdzfkqcscickeabjaoutesvlpopplthufrakrbdhsbuasrqvrfydalelazvbcvcqsacepoqouqqopeybmcohwilxbeqmkfxihmyluexnihndcgpwjllbingqkjtdkjhffxtkastzkjnhmszfqcnypbkgkljjicftjnhvqeciglviiwymqtgxdzmgsbhhwkorahiameuzsurojpiwueddmhaagkghaufmywsdgvvowdeqmrchnkrmipzemfkglslblkjjdjpbensjtbzxvdnvrhupssafrjctzrkorqycmfjmujvrqoikhdkligdkwxyzhsrijskewjmjyzhxvbmillrwlbdqstesbbehnoncxykhfhtdgvbslbdmvlabywwwfsnomjzwuyeynzcmyjbaszxcpeozuubvhbufxaojzynajrzaylqvfmrxnxfdsthsnwwtwsavuzvmyerwfnykcgppjoioqhpwynzpidywlxdboxkoujatbcyqmutrfyrsjaxeoroziuhprntwnjzknpkgztfyfongmuwhcogbqabvvsvptkdmlvivphbrznsjrpprnzfdogrtgyqljoarmioogfiyvqchhmypiqwnltudacqrqiqmjbgviwjoduuhalovvhboumlgfbxvbvfgqikgzyicoaitvtbpsficbcgutbtuxbfqwrdfjwwgoxjjhhtxrcuhknwqrjwfoslypnjonlzglexsfxpqedugjtllqiitflwmqfkfowkhitasbvhowvwrbnxfumkjcbwyzxgfftccptxxeomhnrzatzepkmnyogiepvguvfvfyfmpkjnuufikxwhcvvrjmignitnvjjspxvgjxyqwfdxpszcmzihirrvcxyllhbdrhhoiratcmmgfzejwushbwdziqdhqfsnojgqtkpquiuxmnjyxzxzzkyexkbrgskbylbogbyrdnafioxghllkusvdkgcygdflxiqumezbfcncmodemkerbsxlekkcsqjmkrureosjywrwnfzaoryajlanfouvtuofpymbvjibmdanezzlpgvvokvtcchegfdskyaihgurfjrqaeurlsckeieqbiohefgqxnvwxlhodfxbukafaimjygjdodftdeytymwbemorvrckerwfjdqhgxptlsnhbjybjuerafqhpedrfcgepgobsxbkqttqpdxiymzkbiiyrncvnzxjcksblzvidstnntyssrhwlvizpibjrfvqfacvylcsmcafohldmwojxrtqecsvcgqplpooyutkvhurdfsmeagyrwckqupdvjxczshakmmbtkjwspbtzjqjzpndtegduflwriktdetumspbbitfwdzfwwoyqlqnyqtsxdcdwrofvjfvoojuiauofwsmjkuukcqdohfczyrcyfmbotajdxypbfgtzqsvaausdrurxnbpjdiexvpguifiptbrsrspkdrutbnthvtxbaiglpacaqacrwpbxlmizijfseucvrkhofrnglzdzpksbiestjehvqoqfcefdqsbxkyylszukmmlvsqjzqpnwmtwesfdccdxbndcquaxnuwbnvbpsnwnejmtrxfeglbxrpxfxgedupdxlhbdgzxyxjcfeypautwvvswxuugjeeiehvgfnwjrxxkvmqbnshxbskdlxestjqiyphnizrshmszujopvsjrjambyclvlugylfikdognixfkloabzusfcyhiolmhyshbvoeqdswtusiwapcrbznzgldwaamomvkwkuewpfllmomuhzvwxbickdymcrpezmjnxgtvvmywsnvpbtdzgmauatblqfgwwywftylbvwqmadzvvjtvifqdbupcfmwmaiainpuhsypwmrsgcxyacuiccbwxxypwjceykxxbbqixunblwdzvbohczwayyvyfenfflauuxxvarbruvpkwycuwghxzdeizuwqehgxshzhcbmurkvhkeszqlijcvmgyavhvgtzsdgiptmojswnrtjyehfystwugvlyicggbtscbztahoowxkpprifxixsrdfyohtzwyncjgjpeslukofxtxhlctsixfvpzvlmwoqmhxdjeuodkoshrlwwfnobdkaemiviuyyqedpfiqacitttrjkacklqhpnxbrbdetbukeqzcgvmhrmqeudbrrryzddugmyapmqqepgufhybsobaidjrsqzgamyvoeiqsofjsugrqhserhqzxpzdnncntosgwjdqclftkvkpoyiconsslpmdxxstapfyxdjrgicszltckddxonfwmjdzsdpqoreboirfjkfkenzlktoxakfsaapvqornemrdmgbltzrjnuwnomskunnemmdwtdideytcxcmqnvodwkquqpjqtjibfgvcjsjyvpncuacbhjsplgobpnufotaglmfazhxgkumffkzqszkbigwurmggrwbnmmdpdgsxtxivucijynfbxijwairmyfjjcuglnzrgkzonltxnpfujpvyevupcnrawumhqhapeiyouniqcyatmgrzsyaoxejsxaeerucumnahrfukyxpcznvnnxlnvctnxupvrtzypummothmbyvrnkghdqqdpfqxfxgdtnqyjbmdpzjmmmdgttuldpgjasumifjaluasojihpmcgmlvztefjfpgxmyliwfztkfmmevbyzsdxiixpnqbqspuanabznxtjzekgmkzlzjoycoexfjaoqapkknptgplsldzwfazuvdxqahaqijittbgqdrnhxgpokfezbawsiwumafehzppautuogvsahkfamrdeepovlvmprmzzrkpwmweijthqmgbydwonpiaydrvpcykwbvxhmcufmonmvyecsyozzkecjiudhqhuwrqvttsfnfxltftyaezjohquwyyysouasydxebixsplgwcljsljradusoccdvjgincsuydrqxabsfglcfefwzvfqzywpeiucgtrcweykgiwngbcasuwuwgvdjmwbabsoxrqrokxyrykyjuckqiwwaisomotluenwcbqsipjveegebrqxqzugrtcqnsagrdbqmmzffmgkilaizklvmtdgbhrsvvbuvpbcrcmczuaqfnsoxvlnfpawhjdyolysoawziztgjuwtluribermtzlqddnaqjwnhgmhvecwbyqslajxagtjadhjvhvgdovtmtqjrssrhblznkewgtqrfrvholdtlsbntjrylqlphpsmmgupweeecjohjubaofjsbzcsriotfljcmflbbasjmeutrnghbldmxvowgamoejejouvpoqqqahtcluznvwpjuvpfhongubixsmlyxxmpaanurokugbskjdqrkrhbkaktbkvfvoxuzftzbqkpxuvavzqyhawteweguqzyfdnznbfuuwifvxrtbahudcyjvlgroehzbgshqrthnpczyqxwlgmopubhzsbscxqsecxrexyuwyvmutkzvijbolnrvpnljhvxvnzxhdbhktbfakpvchjvjxgljfvtinuuhwvfydfnaghdrsaljmfghjrappzqvrokwljycarotidjiolbvailscqkudeenywudzqkjdyivecnzhfmulodrqwyqrmzzuhjlktksceoelrragvlnbilflkbjydotduabakfejjamcdffknxcnnudlkmsjboessgqdtplhtlowirwkrvessnctcysoalwjvbzxpyxqszbhljvthaphrvrrrxebdgnewwdrumitninqihlhbrihovfcylnxgskdaxmpaklkjvdjtxolchkdmpeijkywwbyzdworwvxogmrpypngqawlnebybzjflbqotjvwtomgerzklfmofsqgrfjcxfupbmxhitcrvoozphxrgchchrckigpashhtwpxedjhbntobvpizbykcylfolblbnwrechfpjhxrsfidhhxliraxwxfqfzocgehglcspmdhyngdqdoazytgokzonpagaaxwvflwuyibbapmojevlxbbpzfmzslqlyjjduwlztvfxblyqcnsfaoexmddugjhtrqostouzcccvevqircusnzhkhkmsgqwzbjwaipqrnkwiorqtvzoigkbpysbvkzvivuxwtjhucgrcsevqtttddluukitfrcdxmbshxmssmdeliazkvqmgsgqayyqfuhlmwbypwqwamaqrbiuwobczbrijwxxvnrkaywpqljfsinzsuczmlzrycvebgmlupoqaytovvncgwakgtvmokcawksniecsywmbmgxqukuowbujxndgcdvhhccbxdfdtupfchckinhqvqmyrgzhnkfnpghwioowhaxvljhytwmaxojaflalojflawefygwqqxsytutujwtbhibawewagxkybusekmtxrpaulrnreqwndqifsggbnorcqbukdmnsdflxecjjebzppljsjnnnpbnuppmpxrflssthotalfxwoaubwaffzplaoaprzmyprfgdjzceihjripqmflplgooqxmmpzjoafjnttsaqgpuambxspuivihxeqbypqhsmrbjtkaektxwitfbhmbqabahriadwmmnkcahnyvpcsvmteisjopmsyfnpdvlegzhuqwcaqzxwwzzdwegutxlgbgcastyimovrmupxlxtxqglldrghginuabfkgdmpfcvbuntalmljhblvwdvlcykzdnrrffbaiffkxyzlgbzvzvlswqcsqybrixlmrupesrnhgemquzrhmhtpexnzknawokoaasobybyfsbrpxfraywfaumvynmjafzimwhxgxgcjrlryiokemhgpaxstgggybjbzlpoblyvmwjsudirqwuboqixvohypyrgwypiwwfllrqopfosltidbzwvigkcdzvuuxjecuygwdiwiuwieaojwfzmtxrttybqilrgxlimbfrwlezzfvygdwoppadtznshehxhmaiputfxeelkjrmbojtnqnhloqeeanulpyfxrxgnzkwsmtifvqkmvskvycgasppakgsugkupvtouwtrclngwqpmrswrhercdpdaoitcoardzwbvftuttvjvncxqgwnufruxptxwjsiwrdvzauklioyakxcokrgpfvhcguzgzckwvhumbegecwgprpbkszoczrpqfkgsfepvbblsuwvheyggosicwzmimdclpdcfkdldzqnibrlpehdbpykhdobeuehbixedbiyhghoxaqiyrwyeugqyqndmlpulxwpavavlnnirolatvrpztrrlqssvqyvycqcvouaqzcjiqknjgmafjuovadsnauhjninhyccvihkldsufjiszbhqbiwbqgyxeqtoxmuxtitzljiwzmieymdxahqwlmnxcnehtyrkjqirruuszaloqbzxgvrvzndhhmfptuwhssclafxzgkhclvqxjeoiprniewtqobtacnffsaxvmfyynrwnrdrvjbnhqaugbkwlordirucuppedcgqyplfftfpzotdtzozglvlytzobcifzzlijlqmpylzajgjmywsvdmehbtlibbdbkhcjvanmnswkasbiiinunmkmwanvoaqodteegrmlnyueerxnthouexswkjpkrfpmzblbycrfculeojmmcfdqurnxhwqnmvjbmvqqjgobubsbflbkzeewsbxvgbhgbyxiqrbecylmbgotwtxebxvppmwbzmegsjjatygluakvfhfkjvtroszbbuvmyfronzazuepsmtudmvocubupodvtbokchcycfywfalgklgqgurypfnkimoixchbqnfsybujgmatkqntmtsvjpkpzuskcfxejrcjoxhrguddnyeizooaaidalerhweqgoluqxzexaailanyrsfnypccimkekgvoahfcynhvfdowxjywkzwwelkspispuhtoxyxacygtbgynvdeiwhgdmacbnzmqtkvxnvzokepdjbqadpkriiucdvdjwyclcikrypvjaysexteyftxrlrthrxgwqgqgayxfrfzzuhjhpmuewkubcitrlnlxrsujmlkxadcizjjrgjhyljuglvkchguieohohtdulxdgvhtoxbvnuogmibtxdsakiomdnylyjuixixsvtuirmljrktgsbnajpzoneunsmqjvxjhzcslmqdyvfiopvhqeetegdbhxpveymraeshykoaljxxiwjkrwkfvsppvmowfyfmrtphrryimsflivaanyeyokyaraocejzgzxgidmbsvombwqsfktzrdikylpojukcwsjumalvvlmcokglqxkfewayboxmdwceygdjpfvbgindutpydluwwmdupathgosoxnhotozldalbmtkbhwahyajrsgvcuxfvpaqqecowhywrdxmjzkasfjqvbumnahuplcjdhxkunakiavmrazweofqftodajueujgeeaylwiycfsycvinwhbtofpdmksjwafyqftbocjhmzmvljsnrseevsuvirkunfumzdlgcuwkymwczpprpgxvekqtxgdibsrwiqvguyikunecbcmfruwajkrloccwphthvcljrnoksgngvaegkrpscazqlrpjlugzzhjrifaucfwhmmmxpersfjvxbhqcuxqwunoweuxfvolmtuflwfmejuywvbukuzlpmmonmkuwyyxbthcruxjplzpzhvwrozrrjwbyfihglrvpzbsueaotfrhluybzhzftqluynwvovjidxiguchsyrhvidbscdhmyuityteugdfheewffmkthamxcviitrpyjsiqwmruhbkxqejizxuzpzsvmvrxoizkhqzzddffbptfsscgziqtnmdlavufcbftkqkiiixjfbvfoxjulfaqiwqwoceigmzhzhmywwucxwtlbsgmvsqzrkaielnkecuznymtoowtyxangznhvhgvjtrynbqcdirgeurzllbkawffzfwdrmpacouolnhvtnijziicbecojcskswolcrlciulxodqetdxzagstitmknechbzvrqrgmdeewybbxjpgwbwzzeaqhdxtfopmshevgkfgehknkefxwqaukkwnfvtccoaxgrgagupepypnzbductjvwpqhmlczhsjemnzpekvweoaeyhilnkfnwinryjjrgmfdjxobhqammpvwnifbzwidkkrjxjeelejmjzfluyxhdozzoikylwdzpijurxmwvrxevbhckhbpkcbpniuffhgxzepvftlehjaugjvopiwooyqyjzmrutxlublxasaehftjqwpqpampjxlogjrpxgglmxyrfchbxvriuinsmuoevwmfapxpskqrofofhwrfwbycxgvdrklriayrkxuyvfdblfyosdeoysmwnutitsyisawayvlrhxmyzqnwctvcjoljaziuuebkktkqlujsdvmtfxjevdhqctwqtbdrokbnieefsrecyulznjbqxoajnqewosaatnwxqrezftlpmyfyasyxojhdsijrbzpjggsrkmrpobaxmzbccpvefdgxmkvstlodakkzsmitdcxditkclmoinjvwlrjepdepuhohhesuzehjisehnufihzdmksigbvqjhzasxrtysdweiaghspogclgkckkpivpctxwtoipbxbpmtnxsporlfpxunrhlzofnidghpvauhlccjulvujwovdfqskwbunimyvkzbalocwjqcmnoaamsxrbodkmgpkoxpwmqjlfsskesmnvjcqdbtdvlwqofwmsvufryeddmxioxldlabammjlwlicdfysufglfyirjaexognrtckvhxcbmbbuxruxzylklgxowstylvbdqxwnxgdutqzjxszrwzbnyermjqhrgdrbyesgoirecoswohkzchnovkvgvdrqheaahmpbadvbuyxcasdxwexsmscfapbookhedlpeegzkjuxjpjpbrpuqzactilgujdrfrxvzxphgkvrjiqiipvbgpskxyulukavvfmjblglhyucfpbwdkegxsjjhwnyxdxkvbplksfzqtrttcvzjdpeyzdjnncdblvkxqrjkzhnwfhsyxtbvigvrxlxmsngigseksuctxrfpvrvamhqttmwuqkwqbifwqbcghlewuatkftuasidvclzwrbwexcgxmchgbodhxvswvdlbymvytmclgabgpnbcdkawrjqhgvweyuxhjdinuzakvhhqjlyiueivpiorbhzjrphzqbtgtelqpfljwdifhthnpdqeswunpzzifvfzjfahhijejrohodvforlqrhzemvkrdmqlevlfjkcgcwcwknfbvdffargdzuzrwejomsfvgovovfzemvtakrwnxweqcqpoljhccpfwrupbseugipbzkcfrukwbmqrjcpjtnayvitpotykaapexjfrnrmhebypdruxuuqmelojadhbaztuhmukyuedhuyweojqnywmhhssbacapuqcsbmeafcfiffmfqctestvnorhjscvumlspjunfiznbqxqdllwpoylnbiifcgscecgzkjmvivrigbyyhmkxjjlflwbhubtfsqkxzwghfbndbkhlbgwkqcnqcjalcdphyerdgtlhgfzhjgowiqrxhtmuyyxrcxlahnbguwvnxgtvstlovpnfzrzbyqwbyjwtlpebzvvgolggyxqnqtuboqpurrqfwfqsxkcuqcaifeezsdqwdlgwmyhvqqkrzokrehfwdoeskgdloywfzjpusfxfraciuyyhmfunepeghgxuqtxhqseqnljjxvlipcerbobssggebhwxxltgiyjfmfafzksdylbezxvdpkebhipxyndzbyscpfovbydwbqobylccawoqfbtlulmoqcfnprlkxgmvidhdtznkfnjwwramsaowpzjorskjhkewtwvhfelvmfnffwuyjtfsfmrxbyrtchqgjathwtqomoiafacczseirxnnneroukshpfvhdcwwsxkowaokwfwijcfqxrzxpurvaubnsinuaqmzgtpjgirrgpqwxbjwtxrzmlagwyqlhtzsjbzsampfjwidlgkkprtezrnmvlhiivvlmqytufmkbmknnavnuwkubnpbreivfcplvubdcqffwhqnscmbfstrxeltkbicbhmpcisewraunztcnbjwztammktjsfflgngvvqntmlsdcovkxvqnxgybqbjsvvnelnxhwxxbojgnwrowlimgetrpbzpdqdsvoxgseusuamjgizqqcnmrykwkuuazthrjfkehiogqsippsrggylllpkcyugslhywtgrfizsjyiufshiwulcgofkhkqgefibnkglodiwuyozxzokspqzwyuwtawhcjgaelbsdlyrjtcrgbbxubateywtmiuxhewpbngqgyeiynwpfzwbffuqxyygpbnsmpqjjmiljzxrntodoyzihbhderzgyyahpukhhsjsmarlciejoazfpwxmvqhepxzkqqyklglbisqvkxeqezrmxgpxrbpauzcihudcffikaoennstouwltrzgucnxvqypognvsqxqujzuwglncnszxwwciccfdsutgsoocokpoieymxbxvjmatfkonjmqpigbpfvkdpuwilnnofivystdrccdkwiuusuguimxkfhjedszhboehfhtaycurlyzxxlbsjxnzjknpnahoiezfhltuvyrrmltacsrvchlnakqbpuqpbpcfxrapjuuhgobqvqrplzaowtzqgcvzxparghblwihjfmdmelakymzidymslrafdhrrtwjacddniivtgewgbhkvzvocsfsuehxugsukgrybdiynuihsodwdlfkjtgpjbextdbjabxgjvrgnozwikdjxwminopgncsnpghibnfzrsmbrbandudtrjapkzbukatorpiycqpgecwwmezopdysggnhuwgmqejhflarbidjhljxoxppacgdexrhnvgtvrayspmhybrhhtcyfqyrpbvcecbvocfuiroaydnwevngjuwzhojfgoeozkgmzacxvmovauybutgmjmiesevjdrakenvkyykykoogolgvojumtxznbthknlyuunqkcizhxvfvogjgnxbskjyughlgopxyiqbsdqiijecxiqsstsicvxnobshgtomnzomhlqjlqnafipwnejirwmdkpnceoqpnrrpntsydnvuvsqqbbppmuhnxhksgrxzmdiyjcdcbaemlnklldniaxioiepwdgbvplmbdcppxjkvagvdslmztgciinbcsrtjnrtvzcdexvjmpuycyluulrsebqrgorjozzmqoynfnxoduufhwryagvcfkrxndgxnuldqaocpmhuzsxilcpwkhbpcwotynieuezobaogyeklierfclnzjcojgnutcpvwklfonadujbswodtcgpowsaoudceqzskauxirwbdzfiyulmftjmvcedtpqqkrqaraqwtcjmacnyhtibrcciwpttjszhvtsdcjtadfubnlqvcqtudkasmhmyihrlmrlhradplgoshdgzzxqkyrnelpiilotohnlznowtbzdrlbraaijmnjmhcvroarlubpbpyypqmqgrxckynyyqclnidwybiszqluufrkiteankefswfwfqbniuhywracoppthdentqzrojcummolhuwvwxfdsoadbimkutosqhnihhqedtlnhrvqfnubampcbuwfqadsekfczypojvbtzmksmsjfeklmszxplqixzcvmgpmqhdehmmqyybjhasinexxswxyzoodalkwnymfhzgtcyvrlfvpxorgeikzwsfamuqonwkvtkajnynjscczqvgkimngzzkxzjqdydnzsqhzsmlocvewvcgilccuszzeeldbxuwpvgrralpigfscfisrgqbkvmzeivfviiyaakjprjwqkjqsnghwngjmxdvhrqzrojuzxkbqwaqarngbtfgnahyajpqknpwnfmpqwppjgjcsbejlgskdhhqkkefxvsruburjpdvkeacbtdnmrnzkgciwzlvtmetbjjlgynhudqzruzfexqopgvawodhkisedjzjztvnkyzxcnraohfrxrntwyuuvxdarxzthzfzezpfcmkirijiwxaujebmwlpiutbyheipgewfuolbvqhkllrjkdhspzkhajwmzincyhcgmcyjxlmahgieyypdzowzzghwrfpyjosnzkeaskqmzigxvnyaddnwuwyyeawwfmbolmshdmlrmbwffrpzmqrkdevwtdwpwftqkakjtvpgqqzpebtzdrdcjlwhylbqjhcctcypeqtxfvfmjwqwuemlmtwscgbznlqipskwntzxdwfickgldmcnkfesunaxszkhnbbxzwubplxzfhqaajhdsltimcyjdwkgsomurzorbjgqsyuzngmfwrbakdjhdruxijpngjzkdczwstzsldvapzwvjcagyiiukngbmieuoxfksuiudgmsawuejdvpvzrochvxkdllmgvyyjwklzsisyrsaqohcolfwasmjnpsmkkbkgteozbbqigmpgsuuevudqrvzgjfcblmjejknxdsliexlumygsxxeovxyinrcuucrpwwsuphxqaqkfrjyoluoackaejhcatdsxehcqyuitcsngjnkresouxajjavtgdvhavayegpcyilgthvhyzkxnxwosvcxnjnqkyxrxvdnanxfbcisorgbgtlhxcbbptokbehmnmaicvjqqqvuyibqszthpnhgpvzqspmwkmmzffwlwdsvdtvldzxskllurverhwyscljzpasmfshpypcczhnuycupmqxxlkihovxwgsmiteuyviqujyoozcucsgkkpaztvirxctfvkrayhpaespfknfskotubenfwnsyhibbheafyzdaetwbthnltfaixnjccmumglhaotbayilsoqbzkqomwpeumknubhqpqqgxgtybnamitpjzzkcuvrdrmcjzsosjlippecnqwssbesjigfxnosslewcrmhaotvqhayycfasnmvbeinbuovkdykfjchvyrrnhqynzmhgjtdakkmylsgdyobzzpvcpovfkkdwjkyrmtgsqhcfxcmbtcqfqgmtbwlhjxqbhpsysrravpuuphcpmavftyewcrpnzjcmzocynpalgxbjarrziyeyjjijcuogeoxszljkmvjeiymthuflxgmakcitajmowyvqutwrtkkvvcakuntbxdjbczjfbzghotahbctgaubxnkipmbseflhpjuaycpctffyshkzolxcqmhvdaazvznuxewhcxvranesvhijzuroeqedkuizgisadoicwrzrhgakthxbncpjhsimcyfimqxvycetdfockpwdlqwwdrnoermqppigjxximxywyqkkwkwaucasnwcklwixlqwzqtjnwphikojrvjqlojoaneedebegpfjftqkayyxpuhrefkhxirylecbneczhzrshwjlzhwxraxomeeoeuwwbrzdfoapyrqtfnapgjlcraqkxknxmdzaeroqxxdfiojfhriglkznbicjvtkbiogzudnhbwyjdeyedkfhspitgraihcdacjmubvnjdnwanvzuxptapmvsshxpcedeexxresraraugmyufvlgvsytyyoywyxyglliyunvyzjscfomcihbhhdjzgkgeqmmoljyharlqtkebsynkhpchvmnaqowokkeyfbjlhgpltiadcimihsynghjukazexwlekcvmbysogtwyrmdqltgjywncvryhilrykwklxawdsdirywbcxxchwjtqmwnzbxlinbmizoreloqlmmxvyinxqooabhurdtjbmzbkitfcybxcuzxetqdzyhsxyfhuumrskczbhppdebwolijwxezfhuxchgzjlkpbvdtjlhzjwvessbacxoykbhpzsyuopqeygpjhvfpsrnucvwqektefyijqwguyvhokhrclmfrcbqgfbwmptwlxxlevwijkzjoxyzpyrvpslgcluqqnudesgtmhivigoqvvauurobglszvgfmsvcpgvfsbrpxoxekanfnsgnaponjvskdbeedbvbqksmefpkkwpyxwljogqulnemkjehkygehxiazmxnccappvohjdyjuaavoashzqzzunzvuphqiejldbszupmxzwmetmhvxfoekclksdycxmqvjvjlnbwyspzxsjchyzchlqmthjaieghcbshcdatxffralzxqxpgmzchakrtlmcsxfyjeybpuumtpjbxoapbybyvxpoxxndlfgdyhpglxitnoxblqhynsftxcawajlrrhjzbbtfkabpokrwwylbtbxvmclikxhugbqhysmqrybgougphtzynthoyeoimjhhobisjrrtxzobrrwmuabfyyrfaegivpreraccxfzfsaineogdlanjsekritjeimmhkfphgmefyvxvoffccdnxkkxulnbdwkvsyenaapohridltdddlqlwhupoevrotahdplzgqczisxfrfxupyjllyujksarpsefnyxsepgeecwozmlnywvglliysvlexsdihjdtkzyfwclxbeeybggkihorqymdrozvcrrxkyvqpyjoglwscykbqbjapcovxfssfeyvyergidholownvlyjwnnqiyxpqgtzrvplpsvzqugajgqgpimsvqtdobpirwxtdrzqzsjaxnujyvteraailkwlokxwjdkkwefluplwnfstwmfftltbnfkusjuqyszirderbngyegrsuhuovgmscttvqszyrstfokbkarkgrvwasvjusgnmuncekuzcszyizehuuckqnvxahpwdduqituewlisjqvictgmjzkochdmmrzmxmaovrhxlwjrqbsxbkwlvfniuudjorfanzmzyimzdpbcgwcvjtfiyzvhkahacsornscrcfpeedptkojnpgdbgjssgdipvgwpepatqxuuoqfdesazbrlrmsosiihnteomconqelnjigyulrzoquezrgdsqudzadrtvtnajhmsssnulcuptbmjfnpoflwrpgxwdqzxzsrydggzqetiivylpjpdgocegzkdwhkmpkztsryfloalzgpfopwmgxyiibgglrwzddnbgjuicbrjenqfhqeubfbamudugybopzwflnoejothqcixgvmozhxdbijuhnfiphpdmojaepamgeirobagwtaiizwzdsszzejwxppzqpotcsbkeixjzqjusxnmuzkfbymgahibeffuzbrvjslrxerijwsauaqqqfvbwtondwdugthnzkoulgykqkgkyljgtmdgjstoklxqdkhfjolakirwodvqmxgjlxisgkvpcmfuxuwzhltxorqvwwqjqyvfwvibgswsvnjwapsionfpkpsbqzkyzrwhbjlnvovblufmaoangzuuestyedciukuagcyqmlpctlnrhcfegcviwhrlrrwqbvrodaaiqezodbnctszmenvkxnmtqsotsusrinbobtnoguodrjcebzsteucsetmcoctlejowwpucqgyoctbznssbozwcrrhfmggocefczcdpftxcksitfrheoyeohgmuueweyiqfmzlunqreoshjpndxjfihzfngbiyxtyydkpwdfcdhvxskncvyjundpsmpiiqjfkqsdxsuwzsgjbnzgdipvwrhhsppxbmvwrdcllwmlscmqxzephuwbrlumclcmzeakinlnbpxmqtxrwajwlosxibdqlyjwmhxlkbxvjqcghhyvfknporrzjfsmxwcdouzxwxbxnzhcpjnuledsnhpiuxtpetuxphvmgfpsitjzfvdivwqbgvncsgmbkjzwoseuhjzrilvqyezfagtxevzrsjbevsuqqcafsntnayzboyoznucljywgqnochzdfqjqqtclvnaegsirqrdzbyozupzhfbtypytcwxubeazndutrbclakmgszhhpgrwmzybuohhnfxoccknsuarupqacbbvhgtctzpsdtbkgthttwvjzzqsispjlivflqpijdhjpkoumlssqrfvfrvkreubstrothhdqnlbkuavqxklcigrjdvpvrzxcnsgvvacgitswrxazcpmktrghzkcbccvpefzwuvrnbpikqahizyaeufgfkqlofgfgugugejtqsulxaxsrrwrteebumbbowihewbkwvkdubthpxwpwdbvsebxgdovxzhkaypsvjandfwfuwhfqluszcqomlwsjrggbemcefttmffejosvznywcgbemosilcaqfehnmqqzbmjbxglzmsjxqdlppjpwajajjzayirgyxcprghelguxnpbhmkrraqjbyipteuiazvjhhwyoumlazsrxanpmmvxdqecoelezgvvhqunpsdlipuoyfjzgbstiswyizfennyrhfsdwsktgxjefzluidjdfyewmznagjitfbksljkblcidbmcvriykobjefrijcdmgtmtrzcwzxjvcghppymy"));
//        TreeNode treeNode = new Solution().recoverFromPreorder("1-401--349---90--88");
//        int[][] nums = {new int[]{5, 4}, new int[]{6, 4}, new int[]{6, 7}, new int[]{2, 3}};
//        new Solution().maxEnvelopes(nums);
//        TreeNode r = getTreeNode();
//        new Solution().flatten(r);
//        Solution solution = new Solution();
//        TreeNode root = getTreeNode();
//        solution.dfs(root);
//        System.out.println();
//        solution.dfsPreorderWithoutRecursion(root);
//        System.out.println();
//        solution.dfsInorderWithoutRecursion(root);
//        System.out.println();
//        solution.dfsPostorderWithoutRecursion(root);
        ListNode a = fromArray(new int[]{1, 4, 5});
        ListNode b = fromArray(new int[]{1, 3, 4});
        ListNode c = fromArray(new int[]{2, 6});
        ListNode d = fromArray(new int[]{3, 7});
//        ListNode mergeTwoListNodes = new Solution().mergeTwoListNodes(a, b);
        ListNode res = new Solution().mergeKLists(new ListNode[]{a, b, c, d});
    }

    private static TreeNode getTreeNode() {
        TreeNode r = new TreeNode(1);
        TreeNode l1 = new TreeNode(2);
        TreeNode l2 = new TreeNode(3);
        TreeNode l3 = new TreeNode(4);
        TreeNode r1 = new TreeNode(5);
        TreeNode r2 = new TreeNode(6);
        TreeNode r3 = new TreeNode(7);
        TreeNode r4 = new TreeNode(8);
        r.left = l1;
        r.right = r1;
        l1.left = l2;
        l1.right = l3;
        r1.left = r2;
        r2.left = r3;
        r2.right = r4;
        return r;
    }


    private static void testSortAlgorithm(int[] testArray, Consumer<SimpleComparableVal[]> sortAlgorithm) {
        long start = System.currentTimeMillis();
        SimpleComparableVal[] a = toComparableArray(testArray);
        sortAlgorithm.accept(a);
        System.out.println(sortAlgorithm.toString() + " cost " + (System.currentTimeMillis() - start) + "ms  " + sorted(a));
    }
}


