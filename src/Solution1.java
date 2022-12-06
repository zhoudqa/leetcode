import java.util.HashSet;
import java.util.Set;

public class Solution1 {


    public int numDifferentIntegers(String word) {
        Set<String> integers = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (final char c : word.toCharArray()) {
            if (c >= 48 && c <= 57) {
                sb.append(c);
            } else if (sb.length() != 0) {
                integers.add(parseNumber(sb.toString()));
                sb = new StringBuilder();
            }
        }
        if (sb.length() != 0) {
            integers.add(parseNumber(sb.toString()));
        }
        return integers.size();
    }

    private String parseNumber(String str) {
        int index = 0;
        while (index < str.length() && str.charAt(index) == 48) {
            index++;
        }
        return index == str.length() ? "0" : str.substring(index);
    }

    //给你一个n x n矩阵matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
    //请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
    //你必须找到一个内存复杂度优于O(n2) 的解决方案。
    //n == matrix.length
    //n == matrix[i].length
    //1 <= n <= 300
    //-109 <= matrix[i][j] <= 109
    //题目数据 保证 matrix 中的所有行和列都按 非递减顺序 排列
    //1 <= k <= n^2
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];
        //找到left到right之间的数，这个数为第N小，如果N<=k,则向左继续二分，否则向右继续二分
        while (left < right) {
            //中间数，不一定在数组中
            int mid = left + (right - left) / 2;
            if (inLeft(matrix, k, n, mid)) {
                //mid指针往左
                right = mid;
            } else {
                //mid指针往右
                left = mid + 1;
            }
        }
        return left;

    }

    private boolean inLeft(int[][] matrix, int k, int n, int mid) {
        //从左下角开始遍历
        int i = n - 1;
        int j = 0;
        int smaller = 0;
        while (i >= 0 && j <= n - 1) {
            if (matrix[i][j] <= mid) {
                //比mid小的在他上面，个数要+1，然后往右移动
                smaller += i + 1;
                j++;
            } else {
                //减小matrix[i][j]看看
                i--;
            }
        }
        return smaller >= k;
    }

    public static void main(String[] args) {
        final Solution1 solution = new Solution1();
//        solution.numDifferentIntegers("0a0");
        solution.kthSmallest(stringToMatrix("[[1,3,5],[6,7,12],[11,14,14]]"), 3);
    }

    public static int[][] stringToMatrix(String s) {
        final String[] elements = s.substring(2, s.length() - 2).split("],\\[");
        final int h = elements.length;
        final int w = elements[0].split(",").length;
        int[][] res = new int[h][w];
        for (int i = 0; i < elements.length; i++) {
            final String[] split = elements[i].split(",");
            for (int j = 0; j < split.length; j++) {
                res[i][j] = Integer.parseInt(split[j]);
            }
        }
        return res;
    }

}
