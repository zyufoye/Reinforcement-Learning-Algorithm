import numpy as np
import random

def get_state(row,col):
    if row == 3 and col == 10:
        return 'terminal'
    if row != 3 :
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    return 'trap'

# print(get_state(0,0))

def move(row,col,action):
    if get_state(row,col) == 'trap':
        return row,col,0
    if action==0:
        row-=1
    if action==1:
        row+=1
    if action==2:
        col-=1
    if action==3:
        col+=1

    row = max(0,row)
    row = min(3,row)
    col = max(0,col)
    col = min(11,col)
    rewards = -1
    if get_state(row,col) == 'trap':
        rewards = -100
    return row,col,rewards
#print(move(0,0,3))
Q = np.zeros([4,12,4])

def get_action(row,col):
     if random.random() < 0.1:
         return random.choice(range(4))

     return Q[row,col].argmax()

# Q[0][0][0]=0
# Q[0][0][1]=1156
# Q[0][0][2]=156
# Q[0][0][3]=0
# print(Q[0,0].argmax())
# print(get_action(0,0))
# 返回最大数的索引 1

# 更新分数
def get_update(rol,col,action,reward,next_rol,next_col,next_action):
    value =Q[rol,col,action]
    target = 0.9 * Q[next_rol,next_col,next_action]
    target += reward

    update = target -value

    update*=0.1

    return update

# print(get_update(0, 0, 3, -1, 0, 1, 3))

# 开始训练
def train():
    for epoch in range(1500):

        row = random.choice(range(4))
        col = 0

        action = get_action(row,col)

        reward_sum = 0

        while get_state(row,col) not in ['terminal','trap']:
            next_row,next_col,reward = move(row,col,action)
            reward_sum+=reward

            next_action = get_action(next_row,next_col)

            update = get_update(row,col,action,reward,next_row,next_col,next_action)

            Q[row,col,action] +=update

            row = next_row
            col = next_col
            action = next_action

        if epoch % 100 == 0:
            print(epoch,reward_sum)
            print(Q)

train()

def show(row,col,action):
    graph=['□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
        '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□', '□',
        '□', '□', '□', '□', '□', '□', '□', '□', '□', '○', '○', '○', '○', '○',
        '○', '○', '○', '○', '○', '❤']

    action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]

    graph[row * 12 +col] = action
    graph = ''.join(graph)
    for i in range(0, 4 * 12, 12):
        print(graph[i:i + 12])
# show(1,1,3)

from IPython import display
import time

def test():

    row = random.choice(range(4))
    col = 0

    for _ in range(200):
        # 获取当前状态，如果状态是终点或者掉陷阱则终止
        if get_state(row, col) in ['trap', 'terminal']:
            break
        # 选择最优动作
        action = Q[row, col].argmax()

        # 打印这个动作
        display.clear_output(wait=True)
        time.sleep(0.1)
        show(row, col, action)

        # 执行动作
        row, col, reward = move(row, col, action)
#test()