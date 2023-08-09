#!/usr/bin/env python3
from github import Github
import csv
import os
import requests

# Create a Github instance using an access token
g = Github(os.environ["GITHUB_TOKEN"])


def main():
    process_repos()


def process_repos():
    # Read txt files with owner/repo

    repos = read_fileobj("datasets/raw/data.txt")
    malware_repos = read_repos("datasets/raw/source_finder.csv")

    # Query the GitHub API to get repo stats

    benign_rows = get_repo_stats(repos, 0)
    malware_rows = get_repo_stats(malware_repos, 1)

    # rows = benign_rows.append(benign_malware_repos)

    header = [
        "Repo",
        "Author",
        "Star",
        "Fork",
        "Watch",
        "Creation",
        "Last Update",
        "Language",
        "Topic",
        "Family",
        "Platform",
        "has_malware",
        "open_issues",
        "closed_issues",
        "issue_comments",
        "open_pulls",
        "closed_pulls",
        "comments",
        "contributors",
        "subscribers",
        "commits",
        "branches",
        "labels",
        "author_email",
        "author_account_created_at",
        "author_account_last_update",
        "author_contributions",
        "repos_created_by_author",
        "repos_starred_by_author",
        "repos_watched_by_author",
        "author_followers",
        "author_following",
        "author_orgs",
        "author_subscriptions",
    ]

    write_csv("datasets/extend/benign_repos.csv", header, benign_rows)
    write_csv("datasets/extend/malware_repos.csv", header, malware_rows)
    # write_csv('datasets/extend/combined.csv', header, rows) -- fails for unknown reasons


def read_fileobj(filename):
    with open(filename, "r") as fileobj:
        return [line.rstrip("\n") for line in fileobj]


def read_repos(filename):
    repos = []
    with open("datasets/raw/source_finder.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            owner_and_name = row["Author"] + "/" + row["Repo"]
            repos.append(owner_and_name)
    return repos


def write_csv(filename, header, rows):
    with open(filename, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_repo_stats(repos, malware):
    rows = []
    for r in repos:
        try:
            repo = g.get_repo(r)
        except requests.exceptions.RequestException as re:
            print(re)
            continue
        except Exception as e:
            print(f"cannot get data from host {e}")
            continue

        print(repo)

        # Skip private repos
        if repo.private == True:
            r.next()

        # SOURCE FINDER FIELDS
        try:
            repo_name = repo.name
            owner_name = str(repo.owner).split('"')[1]
            stars = repo.stargazers_count
            forks = repo.forks
            watchers = repo.watchers_count
            created_at = repo.created_at
            last_updated = repo.updated_at
            languages = repo.get_languages()
            topics = repo.get_topics()
            family = ""
            platform = ""

            # ADDITIONAL REPO STATS

            open_issues = repo.get_issues(state="open").totalCount
            closed_issues = repo.get_issues(state="closed").totalCount
            issue_comments = repo.get_issues_comments().totalCount
            open_pulls = repo.get_pulls(state="open").totalCount
            closed_pulls = repo.get_pulls(state="closed").totalCount
            comments = repo.get_comments().totalCount
            contributors = repo.get_contributors().totalCount
            subscribers = repo.subscribers_count
            commits = repo.get_commits().totalCount
            branches = repo.get_branches().totalCount
            labels = repo.get_labels().totalCount

            # ADDITIONAL AUTHOR STATS

            author_email_hash = hash(repo.owner.email)
            author_account_created_at = repo.owner.created_at
            author_account_last_update = repo.owner.updated_at
            author_contributions = repo.owner.contributions
            repos_created_by_author = repo.owner.get_repos().totalCount
            repos_starred_by_author = repo.owner.get_starred().totalCount
            repos_watched_by_author = repo.owner.get_watched().totalCount
            author_followers = repo.owner.get_followers().totalCount
            author_following = repo.owner.get_following().totalCount
            author_orgs = repo.owner.get_orgs().totalCount
            author_subscriptions = repo.owner.get_subscriptions().totalCount
        except requests.exceptions.RequestException as re:
            print(re)
            continue
        except Exception as e:
            print(f"cannot get data from host {e}")
            continue

        # GROUND TRUTH LABELS

        has_malware = malware

        row = [
            repo_name,
            owner_name,
            stars,
            forks,
            watchers,
            created_at,
            last_updated,
            languages,
            topics,
            family,
            platform,
            has_malware,
            open_issues,
            closed_issues,
            issue_comments,
            open_pulls,
            closed_pulls,
            comments,
            contributors,
            subscribers,
            commits,
            branches,
            labels,
            author_email_hash,
            author_account_created_at,
            author_account_last_update,
            author_contributions,
            repos_created_by_author,
            repos_starred_by_author,
            repos_watched_by_author,
            author_followers,
            author_following,
            author_orgs,
            author_subscriptions,
        ]
        rows.append(row)
    print(rows)
    return rows


if __name__ == "__main__":
    main()
