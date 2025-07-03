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
wolf_memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=wolf_db_file),
)
seer_memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=seer_db_file),
)

class VillagerAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name = name,
                         description= f"You are a {role},your name is {name},you can`t let other know you are wolf, you can kill one villager each night.",
                         memory=public_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        return target.name

    def discuss(self)->RunResponse:
        return self.run("谁最有可能是狼人？")

class SeerAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name = name,
                        description= f"You are a {role},your name is {name}",
                        memory=seer_memory,
                        model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        return target.name

    def night_action(self)->Agent:
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        return target
    def discuss(self)->RunResponse:
        return self.run("谁最有可能是狼人？")

class WolfAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name = name,
                         description= f"You are a {role},your name is {name},you can`t let other know you are wolf, you can kill one villager each night.",
                         memory=wolf_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        self.env = None

    def decide_vote(self)->str:
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        return target.name

    def night_action(self)->Agent:
        victims = [agent for agent in self.env.agents if agent.role != 'Wolf']
        victim = random.choice(victims)
        return victim

    def discuss(self)->RunResponse:
        return self.run("谁最有可能是狼人？")

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
            self.broadcast_public(response.content)
            vote[agent.name] = 0
        # Agents decide votes
        for agent in self.agents:
            vote[agent.decide_vote()] += 1
        # Find the agent with the most votes
        most_voted = max(vote, key=vote.get)

        # For simplicity, randomly eliminate someone voted most (stub)
        self.eliminate(most_voted)

    def night_phase(self):
        self.broadcast_public("Night falls.")
        # Wolves decide kill
        victim = None
        for agent in self.agents:
            if agent.role == 'Wolf' and victim is None:
                victim = agent.night_action()
                self.broadcast_private(f"{agent.name} kills {victim.name}.")
            elif agent.role == 'Seer':
                seer_target = agent.night_action()
                # Seer sees the role of the target
                if seer_target.role == 'Wolf':
                    self.broadcast_seer(f"{agent.name} sees {seer_target}'s role as Wolf.")
                else:
                    self.broadcast_seer(f"{agent.name} sees {seer_target}'s role as Villager.")

        self.eliminate(victim.name)
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
        return WolfAgent(name=f"Wolf_{id}", role=role)
    elif(role == 'Villager'):
        return VillagerAgent(name=f"Villager_{id}", role=role)
    elif(role == 'Seer'):
        return SeerAgent(name=f"Seer_{id}", role=role)
    else:
        return None

if __name__ == '__main__':
    random.shuffle(roles)
    agents = [createAgent(role,i) for i, role in enumerate(roles)]
    agents = agents
    env = GameEnvironment(agents)
    env.run()
