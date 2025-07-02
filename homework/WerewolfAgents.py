import random
from agno.agent import Agent, Memory, Message
from agno.team import Team


# Define roles
roles = ['Villager', 'Werewolf', 'Seer', 'Werewolf', 'Villager', 'Villager']
state = {}
# Memory types
public_memory = Memory(name='public', shared=True)
private_memory = Memory(name='private', shared=False)
seer_memory = Memory(name='private', shared=False)

class HostAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name)
        self.role = role
        # register memories
        self.register_memory(public_memory)

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


class VillagerAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name)
        self.role = role
        # register memories
        self.register_memory(public_memory)

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
        super().__init__(name)
        self.role = role
        # register memories
        self.register_memory(public_memory)
        self.register_memory(private_memory)

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

class WerewolfAgent(Agent):
    def __init__(self, name, role):
        super().__init__(name)
        self.role = role
        # register memories
        self.register_memory(public_memory)
        self.register_memory(private_memory)

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

class GameEnvironment(Team):
    def __init__(self, agents):
        super().__init__()
        self.agents = agents
        for agent in agents:
            agent.env = self

    def broadcast_public(self, content):
        msg = Message(content=content, memory=public_memory)
        for agent in self.agents:
            agent.observe(msg)

    def broadcast_private(self, content):
        msg = Message(content=content, memory=private_memory)
        # only wolves see
        for agent in self.agents:
            if agent.role == 'Werewolf':
                agent.observe(msg)

    def day_phase(self):
        self.broadcast_public("Day begins. Discuss!")
        # Agents decide votes
        for agent in self.agents:
            agent.decide_vote()

    def night_phase(self):
        self.broadcast_public("Night falls.")
        # Wolves decide kill
        for agent in self.agents:
            agent.night_action()
    def game_end(self) -> str:
        # Check if game ends
        werewolves = [agent for agent in self.agents if agent.role == 'Werewolf']
        villagers = [agent for agent in self.agents if agent.role != 'Werewolf']
        if not werewolves:
            self.broadcast_public("Villagers win!")
        elif len(werewolves) >= len(villagers):
            self.broadcast_public("Werewolves win!")
        else:
            self.broadcast_public("Game continues...")
    def run(self):
        self.broadcast_public("Game start!")
        # Simple one day-night cycle
        self.day_phase()
        self.night_phase()
        self.broadcast_public("Game ends.")

def createAgent(role:str)->Agent:
    if(role == 'Werewolf'):
        return WerewolfAgent(name=f"Werewolf_{random.randint(1, 100)}", role=role)
    elif(role == 'Villager'):
        return VillagerAgent(name=f"Villager_{random.randint(1, 100)}", role=role)
    elif(role == 'Seer'):
        return SeerAgent(name=f"Seer_{random.randint(1, 100)}", role=role)
    else:
        return None

if __name__ == '__main__':
    random.shuffle(roles)
    agents = [createAgent(role) for i, role in enumerate(roles)]
    host = HostAgent(name="Host", role="Host")
    agents = [host] + agents
    env = GameEnvironment(agents)
    env.run()
