import random
from agno.agent import Agent, Memory, Message
from os import getenv
from typing import Any, Dict, List, Optional
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb

from agno.models.openai.like import OpenAILike

public_db_file = "public/agent.db"
wolf_db_file = "wolf/agent.db"
seer_db_file = "seer/agent.db"
# Define roles
roles = ['Villager', 'Werewolf', 'Seer', 'Werewolf', 'Villager', 'Villager']
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
        super().__init__(description= f"You are a Villager,your name is {name},you can`t let other know you are wolf, you can kill one villager each night.",
                         memory=public_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        # register memories

    def observe(self, message: Message):
        # All agents see public messages
        if message.memory.name == 'public':
            print(f"{self.name} observes public: {message.content}")
        # Wolf at night sees private wolf chat
        if self.role == 'Werewolf' and message.memory.name == 'private':
            print(f"{self.name} (Werewolf) observes wolf-chat: {message.content}")

    def decide_vote(self):
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        self.env.broadcast_public(f"{self.name} votes against {target.name}")



class SeerAgent(Agent):
    def __init__(self, name, role):
        super().__init__(description= f"You are a seer,your name is {name}",
                         memory=seer_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role
        # register memories

    def observe(self, message: Message):
        # All agents see public messages
        if message.memory.name == 'public':
            print(f"{self.name} observes public: {message.content}")
        # Wolf at night sees private wolf chat
        if self.role == 'Werewolf' and message.memory.name == 'private':
            print(f"{self.name} (Werewolf) observes wolf-chat: {message.content}")

    def decide_vote(self):
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        self.env.broadcast_public(f"{self.name} votes against {target.name}")

    def night_action(self):
        if self.role == 'Werewolf':
            victims = [agent for agent in self.env.agents if agent.role != 'Werewolf']
            victim = random.choice(victims)
            self.env.broadcast_private(f"Kill {victim.name}")

class WolfAgent(Agent):
    def __init__(self, name, role):
        super().__init__(description= f"You are a wolf,your name is {name},you can`t let other know you are wolf, you can kill one villager each night.",
                         memory=wolf_memory,
                         model=OpenAILike(id="o3-mini",
                            api_key=getenv("OPENAI_API_KEY"),
                            base_url=getenv("OpenAI_API_BASE"),)
                         )
        self.role = role

    def observe(self, message: Message):
        # All agents see public messages
        if message.memory.name == 'public':
            print(f"{self.name} observes public: {message.content}")
        # Wolf at night sees private wolf chat
        if self.role == 'Werewolf' and message.memory.name == 'private':
            print(f"{self.name} (Werewolf) observes wolf-chat: {message.content}")

    def decide_vote(self):
        # Example: random vote among non-self
        others = [agent for agent in self.env.agents if agent.name != self.name]
        target = random.choice(others)
        self.env.broadcast_public(f"{self.name} votes against {target.name}")

    def night_action(self):
        if self.role == 'Werewolf':
            victims = [agent for agent in self.env.agents if agent.role != 'Werewolf']
            victim = random.choice(victims)
            self.env.broadcast_private(f"Kill {victim.name}")

class GameEnvironment():
    def __init__(self, agents: List[Agent]):
        # super().__init__()
        self.agents = agents
        for agent in agents:
            agent.env = self

    def broadcast_public(self, content):
        msg = Message(content=content, memory=public_memory)
        for agent in self.agents:
            agent.observe(msg)

    def broadcast_private(self, content):
        msg = Message(content=content, memory=wolf_memory)
        # only wolves see
        for agent in self.agents:
            if agent.role == 'Werewolf':
                agent.observe(msg)

    def broadcast_seer(self, content):
        msg = Message(content=content, memory=seer_memory)
        # only wolves see
        for agent in self.agents:
            if agent.role == 'Seer':
                agent.observe(msg)
    def eliminate(self, name):
        # Remove a dead agent from the game
        self.agents = [agent for agent in self.agents if agent.name != name]
        print(f"{name} has been eliminated.")

    def day_phase(self):
        self.broadcast_public("Day begins. Discuss!")
        # Agents decide votes
        for agent in self.agents:
            agent.decide_vote()
        # For simplicity, randomly eliminate someone voted most (stub)
        eliminated = random.choice(self.agents)
        self.eliminate(eliminated.name)

    def night_phase(self):
        self.broadcast_public("Night falls.")
        # Wolves decide kill
        for agent in self.agents:
            agent.night_action()

    def check_end(self) -> bool:
        # Check if game ends
        werewolves = [agent for agent in self.agents if agent.role == 'Werewolf']
        villagers = [agent for agent in self.agents if agent.role != 'Werewolf']
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
    if(role == 'Werewolf'):
        return WolfAgent(name=f"Werewolf_{id}", role=role)
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
