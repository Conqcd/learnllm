import os
from typing import Any, AsyncGenerator, Awaitable, Callable, List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import chardet

with open("github-trending-raw.html",encoding="utf-8-sig") as f:
    html = f.read()
soup = BeautifulSoup(html, "html.parser")
for i in soup.find_all(True):
    for name in list(i.attrs):
        if i[name] and name not in ["class"]:
            del i[name]

for i in soup.find_all(["svg", "img", "video", "audio"]):
    i.decompose()

with open("github-trending-slim.html", "w",encoding="utf-8-sig") as f:
    f.write(str(soup))



def fetch_html(url):
    with open(url,encoding="utf-8-sig") as f:
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


llm = OpenAI(model="o3-mini", temperature=0.7)


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

async def craw_trending(url:str = "https://github.com/trending") -> List[Dict[str,str]]:
    async with aiohttp.ClientSession() as client:
        async with client.get(url) as response:
            response.raise_for_status()
            html = await response.text()

    soup = BeautifulSoup(html, "html.parser")

    repositories = []

    for article in soup.select("article.Box-row"):
        repo_info = {}

        repo_info["name"] = (
            article.select_one("h2 a")
            .text.strip()
            .replace("\n", "")
            .replace(" ", "")
        )
        repo_info["url"] = (
                "https://github.com" + article.select_one("h2 a")["href"].strip()
        )

        # Description
        description_element = article.select_one("p")
        repo_info["description"] = (
            description_element.text.strip() if description_element else None
        )

        # Language
        language_element = article.select_one(
            'span[itemprop="programmingLanguage"]'
        )
        repo_info["language"] = (
            language_element.text.strip() if language_element else None
        )

        # Stars and Forks
        stars_element = article.select("a.Link--muted")[0]
        forks_element = article.select("a.Link--muted")[1]
        repo_info["stars"] = stars_element.text.strip()
        repo_info["forks"] = forks_element.text.strip()

        # Today's Stars
        today_stars_element = article.select_one(
            "span.d-inline-block.float-sm-right"
        )
        repo_info["today_stars"] = (
            today_stars_element.text.strip() if today_stars_element else None
        )

        repositories.append(repo_info)

    return repositories

def analysis_trending(trending:str)->str:
    async def chat(query)->str:
        return await llm.chat(query)
    response = asyncio.run(chat(RENDING_ANALYSIS_PROMPT.format(trending)))
    return response

tools = []
tools.append(FunctionTool.from_defaults(
    craw_trending,
    name="craw_trending",
    description="fetch the html of url"
))

tools.append(FunctionTool.from_defaults(
    analysis_trending,
    name="analysis_trending",
    description="analysis the trending"
))

workflow = FunctionAgent(
    llm=llm,
    system_prompt="You are a Subscription Agent. When the user provides a website URL.You can crawl the website by using tool craw_trending, and analysis the content by using tool analysis_trending",
    tools=tools
)

# 运行入口，
async def main():
    response = await workflow.run("帮忙查一下github的trending，网址是https://github.com/trending")
    print(response)

asyncio.run(main())