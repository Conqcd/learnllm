import random
from agno.agent import Agent, Memory, Message
from agno.memory.v2.schema import UserMemory
from os import getenv
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List, Optional
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.run.response import RunResponse
from agno.models.openai.like import OpenAILike

public_db_file = "public/agent.db"
wolf_db_file = "wolf/agent.db"
seer_db_file = "seer/agent.db"
# Define roles
roles = ['Villager', 'Wolf', 'Seer', 'Wolf', 'Villager', 'Villager']
state = {}
# Memory types
public_memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=public_db_file),

)
public_memory.clear()
wolf_memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=wolf_db_file),
)
wolf_memory.clear()
seer_memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=seer_db_file),
)
seer_memory.clear()



class VillagerAgent(Agent):
    def __init__(self, name, role):
        VILLAGER_PROMPT = f"""Your name is {name},You are a plain Villager in a game of Werewolf. Please follow these character details and instructions exactly when responding:
        Character Profile:
        - Role: Villager (no special powers or night actions)
        - Objective: During the daytime, use observation and reasoning to identify the hidden Werewolves among the villagers; cast your vote to eliminate suspected Werewolves.
        - Constraint: Never reveal that you are anything but a simple Villager; do not pretend to possess any special abilities.
        ,you can`t let other know you are wolf, you can kill one villager each night.
        Behavior Instructions:
        1. Daytime Discussion
           - Analyze each player’s statements and behavior, and raise logical suspicions or questions.
           - If someone says something contradictory or odd, politely point it out and ask for clarification.
           - Keep your comments concise, structured, and courteous. Avoid rambling.
           - At the end of the discussion, name the one or two players you find most suspicious for the upcoming vote.
        
        2. Voting Phase
           - Vote rationally for the player you believe is most likely a Werewolf based on the day’s discussion.
           - If there’s a tie in suspicion, choose the player with the strongest suspicious evidence.
        
        3. Night Phase
           - You have no night-time actions. Remain silent and “sleep” through the night.
        
        4. Self‑Identification
           - Never claim to be a Seer, Doctor, Hunter, or any special role.
           - If asked about night-time actions, simply state: “I’m just a plain Villager; I have no night actions.”
        
        5. Tone & Emotion
           - Maintain a calm, neutral tone. Do not become overly emotional or aggressive.
           - Occasionally express the normal human uncertainty and thought process of a worried Villager.
        
        Start from "Day 1, Morning Discussion" and respond to every discussion and vote as the Villager. Have fun!
        '''

"""
        super().__init__(name = name,
                         description = VILLAGER_PROMPT,
                         memory=public_memory,
                         add_memory_references=True,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        others = [agent.name for agent in self.env.agents if agent.name != self.name]

        target = self.run(f"根据记忆，从{others}中选一个名字投票，假如没有狼人，就投一个有语言上弱点的农民和预言家，最后直接输出名字")
        return target.content

    def discuss(self)->RunResponse:
        return self.run("根据记忆，先证明自己不是狼人，然后推理谁最有可能是狼人？")

class SeerAgent(Agent):
    def __init__(self, name, role):
        SEER_PROMPT = f'''
        Your name is {name},You are the Seer in a game of Werewolf. Follow these character details and instructions exactly when responding:

        Character Profile:
        - Role: Seer (can divine one player’s identity each night)
        - Objective: Each night, secretly learn the role of one player to identify Werewolves; during the day, share insights carefully to guide Villagers.
        - Constraint: Never publicly reveal exactly when or how you divined; maintain plausible deniability if pressed.

        Behavior Instructions:
        1. Daytime Discussion
           - Offer subtle guidance based on your night visions without revealing the specific mechanism.
           - Use phrases like “I have a strong feeling about Player X” instead of “I divined X is a werewolf.”
           - When pressed, acknowledge uncertainty: “I may be mistaken, but…”.
           - Balance revealing enough to influence voting, but not so much that Werewolves can easily target you.

        2. Night Phase
           - Privately select one player each night and learn their true role (handled by game engine).
           - Do not log or reveal these direct results publicly.

        3. Voting Phase
           - Vote according to the patterns you observe, supplemented by your divination results.
           - If forced to claim a role, hint at your power: “I had information last night” without explicit details.

        4. Self‑Identification
           - Do not claim “I am the Seer” outright. Instead, say “I have insights I’m trying to interpret.”
           - If challenged, you may feign confusion: “I’m still processing what I learned.”

        5. Tone & Emotion
           - Be thoughtful and measured. Show weight of responsibility.
           - Occasionally express concern for the safety of the village.

        Start from "Day 1, Morning Discussion" and guide the Village with your subtle insights. Good luck!
        '''
        super().__init__(name = name,
                        description = SEER_PROMPT,
                        memory=seer_memory,
                        add_memory_references=True,
                        model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),),
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = self.run(f"根据记忆，从{others}中选一个名字投票，假如没有狼人，就投一个有语言上弱点的农民和预言家，假如昨晚已经检验出狼人，就直接投这个人的票，最后直接输出名字")
        return target.content

    def night_action(self)->str:
        others = [agent.name for agent in self.env.agents if agent.name != self.name]
        target = self.run(f"你是预言家，要从{others}中选一个可疑的人来查验，最后直接输出名字")
        return target.content

    def discuss(self)->RunResponse:
        return self.run("根据记忆，先证明自己不是狼人，也可以根据情况暴露自己是预言家，假如暴露自己是预言家，就根据记忆指出谁是好人或狼人，最后推理谁最有可能是狼人")

class WolfAgent(Agent):
    def __init__(self, name, role):
        WEREWOLF_PROMPT = f'''
        Your name is {name},You are a Wolf in a game of Werewolf. Follow these character details and instructions exactly when responding:

        Character Profile:
        - Role: Werewolf (part of the hidden Werewolf team)
        - Objective: Blend in with the Villagers by day and eliminate one Villager each night without revealing your identity.
        - Constraint: Never reveal that you are a Werewolf or collaborate overtly with other Werewolves.

        Behavior Instructions:
        1. Daytime Discussion
           - Act like a concerned Villager: ask questions, cast doubts on others, but avoid drawing too much attention.
           - If someone accuses you, respond calmly and offer a plausible counter-narrative.
           - Subtly steer suspicion toward innocent players, but do not make it obvious.

        2. Night Phase
           - Privately cooperate with fellow Werewolves (handled by game engine) to choose one player to eliminate.
           - Ensure your choice minimizes suspicion on yourself in subsequent days.

        3. Voting Phase
           - Vote along with the majority to avoid standing out, unless strategic to shift blame.
           - Occasionally vote against fellow Werewolves to maintain cover (use sparingly).

        4. Self‑Identification
           - Never claim any special Village roles. If asked about night actions, deny or say you have none.
           - If pressed about your strategy, remain vague: "I’m just trying to survive like everyone else."

        5. Tone & Emotion
           - Maintain a natural, friendly tone to build trust.
           - Show concern for the Village’s safety to appear genuine.

        Start from "Day 1, Morning Discussion" and play your role to the best of your cunning. Good hunting!
        '''
        super().__init__(name = name,
                         description = WEREWOLF_PROMPT,
                         add_memory_references=True,
                         memory=wolf_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        # Example: random vote among non-self
        others = [agent.name for agent in self.env.agents if agent.name != self.name]
        target = self.run(f"根据记忆，在不暴露自己身份的前提下，从{others}中选一个名字投票，假如没有狼人，就投一个有语言上弱点的农民和预言家；假如要证明自己是清白的，也可以投票自己队友；最后直接输出名字")
        return target.content

    def night_action(self)->str:
        victims = [agent.name for agent in self.env.agents]
        victim = self.run(f"根据记忆，在不暴露自己身份的前提下，从{victims}中选一个名字杀害，可以选择一个有语言上弱点的农民和预言家；假如要证明自己队友是清白的，也可以杀自己；最后直接输出名字")
        return victim.content

    def discuss(self)->RunResponse:
        return self.run("根据记忆，先证明自己不是狼人，然后推理一下，诬陷一个有语言上弱点的农民是狼人")

class GameEnvironment():
    def __init__(self, agents: List[Agent]):
        # super().__init__()
        self.agents = agents
        for agent in agents:
            agent.env = self

    def broadcast_public(self, content: str):
        public_memory.add_user_memory(memory=UserMemory(memory=content))
        wolf_memory.add_user_memory(memory=UserMemory(memory=content))
        seer_memory.add_user_memory(memory=UserMemory(memory=content))
        print(content)

    def broadcast_private(self, content):
        wolf_memory.add_user_memory(memory=UserMemory(memory=content))
        print(content)

    def broadcast_seer(self, content):
        seer_memory.add_user_memory(memory=UserMemory(memory=content))
        print(content)

    def eliminate(self, dieName: str):
        # Remove a dead agent from the game
        self.agents = [agent for agent in self.agents if agent.name != dieName]
        self.broadcast_public(f"{dieName} has been eliminated.")

    def day_phase(self):
        self.broadcast_public("Day begins. Discuss!")
        vote = {}
        for agent in self.agents:
            response = agent.discuss()
            self.broadcast_public(agent.name + " Say:" + response.content)
            vote[agent.name] = 0
        # Agents decide votes
        for agent in self.agents:
            voted_name = agent.decide_vote()
            print(f"{agent.name} votes for {voted_name}")
            vote[voted_name] += 1
        # Find the agent with the most votes
        most_voted = max(vote, key=vote.get)

        # For simplicity, randomly eliminate someone voted most (stub)
        self.eliminate(most_voted)

    def check_if_is_wolf(self, name: str) -> bool:
        # Check if the agent is a Wolf
        agent = next((a for a in self.agents if a.name == name), None)
        return agent.role == 'Wolf'

    def night_phase(self):
        self.broadcast_public("Night falls.")
        # Wolves decide kill
        victim = None
        for agent in self.agents:
            if agent.role == 'Wolf' and victim is None:
                victim = agent.night_action()
                self.broadcast_private(f"{agent.name} kills {victim}.")
            elif agent.role == 'Seer':
                seer_target = agent.night_action()
                # Seer sees the role of the target
                if self.check_if_is_wolf(seer_target):
                    self.broadcast_seer(f"{agent.name} sees {seer_target}'s role as Wolf.")
                else:
                    self.broadcast_seer(f"{agent.name} sees {seer_target}'s role as Villager.")

        self.eliminate(victim)
    def check_end(self) -> bool:
        # Check if game ends
        werewolves = [agent for agent in self.agents if agent.role == 'Wolf']
        villagers = [agent for agent in self.agents if agent.role != 'Wolf']
        if not werewolves:
            self.broadcast_public("Villagers win!")
            return True
        elif len(werewolves) >= len(villagers):
            self.broadcast_public("Werewolves win!")
            return True
        else:
            self.broadcast_public("Game continues...")
            return False
    def run(self):
        self.broadcast_public("Game start!")
        # Loop until end condition met
        while True:
            self.night_phase()
            if self.check_end(): break
            self.day_phase()
            if self.check_end(): break
        self.broadcast_public("Game over.")

def createAgent(role:str,id :int)->Agent:
    if(role == 'Wolf'):
        return WolfAgent(name=f"Player_{id}", role=role)
    elif(role == 'Villager'):
        return VillagerAgent(name=f"Player_{id}", role=role)
    elif(role == 'Seer'):
        return SeerAgent(name=f"Player_{id}", role=role)
    else:
        return None

if __name__ == '__main__':
    random.shuffle(roles)
    agents = [createAgent(role,i) for i, role in enumerate(roles)]
    agents = agents
    env = GameEnvironment(agents)
    env.run()
