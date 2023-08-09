#!/usr/bin/env python3
from github import Github
import csv
import subprocess


def main():
    process_repos()


def process_repos():
    # with open("datasets/raw/benign_repos.txt", 'r') as fileobj:
    #     repos = [line.rstrip('\n') for line in fileobj]

    # for repo in repos:
    #     # # cmd_str = "bodegha {repo} --verbose --key GITHUB_TOKEN"
    #     # file_ = open("datasets/bot-detector-output.txt", "w")
    #     # p = subprocess.Popen(cmd_str, stdout=file_, shell=True)

    repo = "Rails/Rails"
    cmd_str = "bodegha {repo} --verbose --key GITHUB_TOKEN"
    file_ = open("datasets/rails-bot-detector-output.txt", "w")
    p = subprocess.Popen(cmd_str, stdout=file_, shell=True)


# def predict_bot(owner_name,repo_name):
#     cmd_str = f"bodegha {owner_name}/{repo_name} --verbose --key GITHUB_TOKEN"
#     p = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, shell=True)
#     print(p.communicate())


if __name__ == "__main__":
    main()
