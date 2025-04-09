import tkinter as tk
from tkinter import messagebox
import random

class TicTacToe:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("井字棋游戏")
        self.window.geometry("400x450")
        self.window.resizable(False, False)
        self.window.configure(bg="#f0f0f0")
        
        # 游戏状态
        self.board = [' ' for _ in range(9)]  # 棋盘
        self.current_player = 'X'  # 玩家为X，AI为O
        self.game_over = False
        
        # 创建标题标签
        self.title_label = tk.Label(self.window, text="井字棋游戏", font=('Arial', 20, 'bold'), bg="#f0f0f0")
        self.title_label.pack(pady=10)
        
        # 创建游戏信息标签
        self.info_label = tk.Label(self.window, text="您的回合 (X)", font=('Arial', 12), bg="#f0f0f0")
        self.info_label.pack(pady=5)
        
        # 创建游戏棋盘框架
        self.board_frame = tk.Frame(self.window, bg="#f0f0f0")
        self.board_frame.pack(pady=10)
        
        # 创建按钮网格
        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(self.board_frame, text="", font=('Arial', 24, 'bold'), 
                                  width=5, height=2, command=lambda r=i, c=j: self.make_move(r*3+c))
                button.grid(row=i, column=j, padx=5, pady=5)
                row.append(button)
            self.buttons.append(row)
            
        # 创建重新开始按钮
        self.restart_button = tk.Button(self.window, text="重新开始", font=('Arial', 12),
                                        command=self.restart_game, bg="#4CAF50", fg="white", 
                                        width=10, height=1)
        self.restart_button.pack(pady=15)
        
        # 难度选择
        self.difficulty_frame = tk.Frame(self.window, bg="#f0f0f0")
        self.difficulty_frame.pack(pady=5)
        
        self.difficulty_label = tk.Label(self.difficulty_frame, text="难度:", font=('Arial', 12), bg="#f0f0f0")
        self.difficulty_label.grid(row=0, column=0, padx=5)
        
        self.difficulty_var = tk.StringVar(value="中等")
        difficulties = ["简单", "中等", "困难"]
        self.difficulty_menu = tk.OptionMenu(self.difficulty_frame, self.difficulty_var, *difficulties)
        self.difficulty_menu.grid(row=0, column=1, padx=5)
        
        # 开始游戏
        self.window.mainloop()
        
    def make_move(self, position):
        """玩家尝试下棋"""
        if self.board[position] == ' ' and not self.game_over:
            # 玩家回合
            self.board[position] = self.current_player
            self.buttons[position // 3][position % 3].config(text=self.current_player, fg="#0000FF")
            
            # 检查游戏是否结束
            if self.check_winner():
                self.game_over = True
                messagebox.showinfo("游戏结束", "恭喜! 你赢了!")
                return
            
            if self.check_draw():
                self.game_over = True
                messagebox.showinfo("游戏结束", "平局!")
                return
            
            # 电脑回合
            self.current_player = 'O'
            self.info_label.config(text="电脑回合 (O)")
            self.window.update()
            
            # 添加小延迟使得电脑思考更真实
            self.window.after(500, self.ai_move)
    
    def ai_move(self):
        """AI移动"""
        difficulty = self.difficulty_var.get()
        
        if difficulty == "简单":
            # 简单难度: 随机移动
            empty_cells = [i for i, cell in enumerate(self.board) if cell == ' ']
            if empty_cells:
                position = random.choice(empty_cells)
        elif difficulty == "中等":
            # 中等难度: 70%使用MinMax, 30%随机移动
            if random.random() < 0.7:
                position = self.get_best_move()
            else:
                empty_cells = [i for i, cell in enumerate(self.board) if cell == ' ']
                if empty_cells:
                    position = random.choice(empty_cells)
        else:  # 困难
            # 困难难度: 总是使用MinMax
            position = self.get_best_move()
            
        # 执行移动
        self.board[position] = 'O'
        self.buttons[position // 3][position % 3].config(text='O', fg="#FF0000")
        
        # 检查游戏状态
        if self.check_winner():
            self.game_over = True
            messagebox.showinfo("游戏结束", "电脑赢了!")
            return
        
        if self.check_draw():
            self.game_over = True
            messagebox.showinfo("游戏结束", "平局!")
            return
        
        # 回到玩家回合
        self.current_player = 'X'
        self.info_label.config(text="您的回合 (X)")
    
    def get_best_move(self):
        """使用极小极大算法找出最佳移动"""
        best_score = float('-inf')
        best_move = 0
        
        for i in range(9):
            if self.board[i] == ' ':
                self.board[i] = 'O'
                score = self.minimax(self.board, 0, False)
                self.board[i] = ' '
                
                if score > best_score:
                    best_score = score
                    best_move = i
                    
        return best_move
    
    def minimax(self, board, depth, is_maximizing):
        """极小极大算法"""
        # 检查是否有获胜者
        if self.check_winner_for_board(board, 'O'):
            return 10 - depth
        if self.check_winner_for_board(board, 'X'):
            return depth - 10
        if ' ' not in board:  # 平局
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = 'O'
                    score = self.minimax(board, depth + 1, False)
                    board[i] = ' '
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = 'X'
                    score = self.minimax(board, depth + 1, True)
                    board[i] = ' '
                    best_score = min(score, best_score)
            return best_score
    
    def check_winner(self):
        """检查当前玩家是否获胜"""
        return self.check_winner_for_board(self.board, self.current_player)
    
    def check_winner_for_board(self, board, player):
        """检查指定玩家在给定棋盘上是否获胜"""
        # 检查所有行
        for i in range(0, 9, 3):
            if board[i] == board[i+1] == board[i+2] == player:
                return True
                
        # 检查所有列
        for i in range(3):
            if board[i] == board[i+3] == board[i+6] == player:
                return True
                
        # 检查对角线
        if board[0] == board[4] == board[8] == player:
            return True
        if board[2] == board[4] == board[6] == player:
            return True
            
        return False
    
    def check_draw(self):
        """检查是否平局"""
        return ' ' not in self.board
    
    def restart_game(self):
        """重新开始游戏"""
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.game_over = False
        self.info_label.config(text="您的回合 (X)")
        
        # 重置所有按钮
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text="")

if __name__ == "__main__":
    game = TicTacToe()