import os
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional
from aiocron import crontab
from pydantic import BaseModel, Field
from pytz import BaseTzInfo

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

import aiohttp
import asyncio
from bs4 import BeautifulSoup

# fix SubscriptionRunner not fully defined
from metagpt.environment import Environment as _  # noqa: F401

with open("github-trending-raw.html") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
for i in soup.find_all(True):
    for name in list(i.attrs):
        if i[name] and name not in ["class"]:
            del i[name]

for i in soup.find_all(["svg", "img", "video", "audio"]):
    i.decompose()

with open("github-trending-slim.html", "w") as f:
    f.write(str(soup))



def fetch_html(url):
    with open(url, encoding="utf-8") as f:
        html = f.read()
    return html


async def parse_github_trending(html):
    soup = BeautifulSoup(html, 'html.parser')

    repositories = []

    for article in soup.select('article.Box-row'):
        repo_info = {}

        repo_info['name'] = article.select_one('h2 a').text.strip()
        repo_info['url'] = article.select_one('h2 a')['href'].strip()

        # Description
        description_element = article.select_one('p')
        repo_info['description'] = description_element.text.strip() if description_element else None

        # Language
        language_element = article.select_one('span[itemprop="programmingLanguage"]')
        repo_info['language'] = language_element.text.strip() if language_element else None

        # Stars and Forks
        stars_element = article.select('a.Link--muted')[0]
        forks_element = article.select('a.Link--muted')[1]
        repo_info['stars'] = stars_element.text.strip()
        repo_info['forks'] = forks_element.text.strip()

        # Today's Stars
        today_stars_element = article.select_one('span.d-inline-block.float-sm-right')
        repo_info['today_stars'] = today_stars_element.text.strip() if today_stars_element else None

        repositories.append(repo_info)

    return repositories


async def fetch():
    url = 'github-trending-raw.html'
    html = fetch_html(url)
    repositories = await parse_github_trending(html)

    for repo in repositories:
        print(f"Name: {repo['name']}")
        print(f"URL: https://github.com{repo['url']}")
        print(f"Description: {repo['description']}")
        print(f"Language: {repo['language']}")
        print(f"Stars: {repo['stars']}")
        print(f"Forks: {repo['forks']}")
        print(f"Today's Stars: {repo['today_stars']}")
        print()


class SubscriptionRunner(BaseModel):
    """A simple wrapper to manage subscription tasks for different roles using asyncio.
    Example:
        >>> import asyncio
        >>> from metagpt.subscription import SubscriptionRunner
        >>> from metagpt.roles import Searcher
        >>> from metagpt.schema import Message
        >>> async def trigger():
        ...     while True:
        ...         yield Message("the latest news about OpenAI")
        ...         await asyncio.sleep(3600 * 24)
        >>> async def callback(msg: Message):
        ...     print(msg.content)
        >>> async def main():
        ...     pb = SubscriptionRunner()
        ...     await pb.subscribe(Searcher(), trigger(), callback)
        ...     await pb.run()
        >>> asyncio.run(main())
    """

    tasks: Dict[Role, asyncio.Task] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def subscribe(
        self,
        role: Role,
        trigger: AsyncGenerator[Message, None],
        callback: Callable[
            [
                Message,
            ],
            Awaitable[None],
        ],
    ):
        """Subscribes a role to a trigger and sets up a callback to be called with the role's response.
        Args:
            role: The role to subscribe.
            trigger: An asynchronous generator that yields Messages to be processed by the role.
            callback: An asynchronous function to be called with the response from the role.
        """
        loop = asyncio.get_running_loop()

        async def _start_role():
            async for msg in trigger:
                resp = await role.run(msg)
                await callback(resp)

        self.tasks[role] = loop.create_task(_start_role(), name=f"Subscription-{role}")

    async def unsubscribe(self, role: Role):
        """Unsubscribes a role from its trigger and cancels the associated task.
        Args:
            role: The role to unsubscribe.
        """
        task = self.tasks.pop(role)
        task.cancel()

    async def run(self, raise_exception: bool = True):
        """Runs all subscribed tasks and handles their completion or exception.
        Args:
            raise_exception: _description_. Defaults to True.
        Raises:
            task.exception: _description_
        """
        i=0
        while True:
            i+=1
            for role, task in self.tasks.items():
                i=0
                if task.done():
                    if task.exception():
                        if raise_exception:
                            raise task.exception()
                        logger.opt(exception=task.exception()).error(
                            f"Task {task.get_name()} run error"
                        )
                    else:
                        logger.warning(
                            f"Task {task.get_name()} has completed. "
                            "If this is unexpected behavior, please check the trigger function."
                        )
                    self.tasks.pop(role)
                    break
            else:
                await asyncio.sleep(1)
            if i>0:
                break

RENDING_ANALYSIS_PROMPT = """# Requirements
You are a GitHub Trending Analyst, aiming to provide users with insightful and personalized recommendations based on the latest
GitHub Trends. Based on the context, fill in the following missing information, generate engaging and informative titles, 
ensuring users discover repositories aligned with their interests.

# The title about Today's GitHub Trending
## Today's Trends: Uncover the Hottest GitHub Projects Today! Explore the trending programming languages and discover key domains capturing developers' attention. From ** to **, witness the top projects like never before.
## The Trends Categories: Dive into Today's GitHub Trending Domains! Explore featured projects in domains such as ** and **. Get a quick overview of each project, including programming languages, stars, and more.
## Highlights of the List: Spotlight noteworthy projects on GitHub Trending, including new tools, innovative projects, and rapidly gaining popularity, focusing on delivering distinctive and attention-grabbing content for users.
---
# Format Example


# [Title]

## Today's Trends
Today, ** and ** continue to dominate as the most popular programming languages. Key areas of interest include **, ** and **.
The top popular projects are Project1 and Project2.

## The Trends Categories
1. Generative AI
    - [Project1](https://github/xx/project1): [detail of the project, such as star total and today, language, ...]
    - [Project2](https://github/xx/project2): ...
...

## Highlights of the List
1. [Project1](https://github/xx/project1): [provide specific reasons why this project is recommended].
...

---
# Github Trending
{trending}
"""


# Role实现
# 对于V0.7 以上的版本，需要把老版本的
# self._init_actions 改为self.set_actions
class OssWatcher(Role):
    def __init__(
            self,
            name="Codey",
            profile="OssWatcher",
            goal="Generate an insightful GitHub Trending analysis report.",
            constraints="Only analyze based on the provided GitHub Trending data.",
    ):
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints)
        self.set_actions([CrawlOSSTrending, AnalysisOSSTrending])
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        # By choosing the Action by order under the hood
        # todo will be first SimpleWriteCode() then SimpleRunCode()
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        result = await todo.run(msg.content)

        msg = Message(content=str(result), role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg


async def wxpusher_callback(msg: Message):
    print(msg.content)


async def trigger():
    # 这里设置了只触发五次，也可以用while True 永远执行下去
    for i in range(5):
        yield Message("the latest news about OpenAI")
        await asyncio.sleep(5)
        #  每隔五秒钟执行一次。
        # 也可以设置为每隔3600 * 24 秒执行一次


# 运行入口，
async def main():
    callbacks = []
    if not callbacks:
        async def _print(msg: Message):
            print(msg.content)

        callbacks.append(_print)

    # callback
    async def callback(msg):
        await asyncio.gather(*(call(msg) for call in callbacks))

    runner = SubscriptionRunner()
    await runner.subscribe(OssWatcher(), trigger(), callback)
    await runner.run()


await main()