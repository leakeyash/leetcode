package archive.array;

import java.util.*;

public class IsValidSudoku {
    public static void main(String[] args) {
        char[][] board = new char[][]{
                {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
                {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
                {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
                {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
                {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
                {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
                {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
                {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
                {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
        };
        System.out.println(new IsValidSudoku().isValidSudoku(board));
    }

    public boolean isValidSudoku(char[][] board) {

        char[][] groups = new char[27][9];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                char value = board[i][j];
                groups[i%9][j%9] = value;
                groups[9+j%9][i%9] = value;
                groups[18+(i/3)*3+j/3][(i%3)*3+j%3] = value;
            }
        }
        for(int i=0;i<groups.length;i++){
            Set<Character> temp = new HashSet<>();
            for (int j = 0;j<groups[i].length;j++) {
                char value = groups[i][j];
                if(temp.contains(value)){
                    return false;
                } else if(value!='.'){
                    temp.add(value);
                }
            }
        }
        return true;
    }

    public boolean isValidSudoku2(char[][] board) {
        int[][] rowNumCounts = new int[9][9];
        int[][] colNumCounts = new int[9][9];
        int[][] matrixNumCounts = new int[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '0' - 1;
                    rowNumCounts[i][num]++;
                    colNumCounts[j][num]++;
                    int matrixIndex = 3 * (i / 3) + j / 3;
                    matrixNumCounts[matrixIndex][num]++;
                    if (rowNumCounts[i][num] == 2
                            || colNumCounts[j][num] == 2
                            || matrixNumCounts[matrixIndex][num] == 2) {
                        return false;
                    }

                }
            }
        }
        return true;
    }
}
