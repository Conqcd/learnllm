import os
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional
from pydantic import BaseModel, Field


from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI
import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

import random

def assign_roles_fn(players: list) -> dict:
    roles = ['werewolf', 'seer', 'witch'] + ['villager'] * (len(players) - 3)
    random.shuffle(roles)
    return dict(zip(players, roles))

assign_roles = FunctionTool.from_defaults(fn=assign_roles_fn,
    name="assign_roles",
    description="分配玩家角色：狼人、预言家、女巫、村民。返回 {player: role}。"
)

# 记录并处理夜间行动：狼人杀人、女巫救或毒、预言家验人
witch_state = {'saved': False, 'poisoned': False}

def night_actions_fn(actions: list, state: dict) -> dict:
    # actions: [{'role':'werewolf','target':'Alice'}, ...]
    result = {'killed': None, 'revealed': {}, 'witch_used': False}
    # 狼人行动
    for act in actions:
        if act['role']=='werewolf': result['killed']=act['target']
    # 预言家行动
    for act in actions:
        if act['role']=='seer': result['revealed'][act['target']] = state['roles'][act['target']]
    # 女巫可救或毒
    for act in actions:
        if act['role']=='witch' and not state['witch_used']:
            if act.get('save'):
                result['killed'] = None
                result['witch_used']=True
            if act.get('poison'):
                result['killed_poison'] = act['target']
                result['witch_used']=True
    return result

night_actions = FunctionTool.from_defaults(fn=night_actions_fn,
    name="night_actions",
    description="处理夜间一轮所有角色行动，返回夜间结果。"
)

# 投票统计
def tally_votes_fn(votes: list) -> str:
    counter = {}
    for v in votes:
        counter[v['target']] = counter.get(v['target'], 0) + 1
    return max(counter, key=counter.get)

tally_votes = FunctionTool.from_defaults(fn=tally_votes_fn,
    name="tally_votes",
    description="统计白天所有投票，返回被投票最多的玩家。"
)


# GM Agent：执行分配、阶段管理、胜负判断
gm_agent = FunctionCallingAgentWorker.from_tools(
    tools=[assign_roles, night_actions, tally_votes],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="你是狼人杀游戏的主持人，管理分发角色、阶段切换、收集行动与判定胜负。"
)

# 各角色 Agent：仅需了解自己的角色与行动
def make_player_agent(role: str):
    prompt = f"你是{role}角色，在游戏阶段到来时执行你的特殊行动或投票。"
    tools = []
    if role in ['werewolf','seer','witch']:
        tools = [night_actions]
    # 所有玩家都可投票
    tools.append(tally_votes)
    return FunctionCallingAgentWorker.from_tools(
        tools=tools,
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=prompt
    )


# 初始化玩家列表与 Agent 实例
players = ['Alice','Bob','Charlie','David','Eve']
roles = {}  # 状态存储
player_agents = {}

# 1. 由 GM 分配角色
assign_result = gm_agent.chat(players)
roles = assign_result  # 保存角色映射
for p, r in roles.items():
    player_agents[p] = make_player_agent(r)

# 2. 游戏主循环：交替执行 天黑 -> 天亮，直至结束
day = 1
alive = set(players)
state = {'roles': roles, 'witch_used': False}

while True:
    # — 夜晚阶段 —
    night_actions_list = []
    for p in alive:
        agent = player_agents[p]
        # 根据角色，获取夜间行动（返回 dict）
        if roles[p] in ['werewolf','seer','witch']:
            action = agent.chat({'role': roles[p], 'state': state})
            night_actions_list.append({'role': roles[p], **action})
    night_result = night_actions_fn(night_actions_list, state)
    # 更新存活列表
    if night_result.get('killed'):
        alive.remove(night_result['killed'])
    if night_result.get('killed_poison'):
        alive.remove(night_result['killed_poison'])

    # — 白天阶段 —
    # 广播夜间结果（可通过 GM Agent）
    gm_agent.chat({'phase':'day', 'night_result': night_result, 'alive': list(alive)})

    # 玩家投票
    votes = []
    for p in alive:
        vote = player_agents[p].chat({'phase':'vote', 'alive': list(alive)})
        votes.append({'voter': p, 'target': vote['target']})
    lynched = tally_votes_fn(votes)
    alive.remove(lynched)

    # 判断胜负：狼人全部被淘汰 or 狼人人数 >= 村人人数
    wolves = [p for p in alive if roles[p]=='werewolf']
    villagers = [p for p in alive if roles[p]!='werewolf']
    if not wolves or len(wolves) >= len(villagers):
        winner = 'Villagers' if not wolves else 'Werewolves'
        break
    day += 1

# 最终结果
print(f"游戏结束。胜利方：{winner}")