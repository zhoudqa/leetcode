import java.util.*;
import java.util.function.Consumer;

public class Solution {


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

    static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
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
                } else if (j == i + 1) {
                    dp[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[i][j] = dp[i + 1][j - 1] && s.charAt(i) == s.charAt(j);
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

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    static class Node {
        char c;
        Node pre;
        Node next;

        public Node(char c) {
            this.c = c;
        }
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

    static class Node1 {
        int val;
        Node1 left;
        Node1 right;

        public Node1(int val) {
            this.val = val;
        }
    }

    //   7
    //  /\
    // 3  9
    ///\ /
//   2 4 6
    //打印697
    //   437
    //   237
    public static void printTree(Node1 root) {

        List<String> res = new ArrayList<>();
        if (root != null)
            printTree(root, "", res);
        res.forEach(s -> System.out.println(reverse(s)));

    }

    private static String reverse(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = s.length() - 1; i >= 0; i--) {
            stringBuilder.append(s.charAt(i));
        }
        return stringBuilder.toString();
    }

    public static void printTree(Node1 root, String cur, List<String> res) {
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

    //俄罗斯套娃信封 fn为包含fn-1的最优解
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

    public void dfs(Node1 root) {
        if (root != null) {
            System.out.print(root.val);
            dfs(root.left);
            dfs(root.right);
        }
    }

    public void bfs(Node1 root) {
        Queue<Node1> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node1 cur = queue.poll();
            System.out.print(cur.val);
            if (cur.left != null) {
                queue.add(cur.left);
            }
            if (cur.right != null) {
                queue.add(cur.right);
            }
        }
    }

    public void bfsWithLevel(Node1 root) {
        if (root == null) {
            return;
        }
        Queue<Node1> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            //遍历同一层的节点而不是每个节点都进入while循环，记住当前队列中的节点数量
            int sameLevelNodeCount = queue.size();
            for (int i = 0; i < sameLevelNodeCount; i++) {
                Node1 cur = queue.poll();
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
        Node1 r = new Node1(1);
        Node1 l1 = new Node1(2);
        Node1 l2 = new Node1(3);
        Node1 l3 = new Node1(4);
        Node1 r1 = new Node1(5);
        Node1 r2 = new Node1(6);
        Node1 r3 = new Node1(7);
        Node1 r4 = new Node1(8);
        r.left = l1;
        r.right = r1;
        l1.left = l2;
        l1.right = l3;
        r1.left = r2;
        r2.left = r3;
        r2.right = r4;
        new Solution().dfs(r);
        System.out.println();
        new Solution().bfs(r);
        System.out.println();
        new Solution().bfsWithLevel(r);
    }


    private static void testSortAlgorithm(int[] testArray, Consumer<SimpleComparableVal[]> sortAlgorithm) {
        long start = System.currentTimeMillis();
        SimpleComparableVal[] a = toComparableArray(testArray);
        sortAlgorithm.accept(a);
        System.out.println(sortAlgorithm.toString() + " cost " + (System.currentTimeMillis() - start) + "ms  " + sorted(a));
    }
}


